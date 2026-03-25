"""Helpers for storing notes in GitHub via the Contents API."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import date
from typing import Any

import httpx

from mlflow_notes_agent.config import get_settings, slugify_summary

USER_AGENT = "mlflow-notes-demo/0.1"


@dataclass(frozen=True)
class NoteSummary:
    name: str
    path: str
    sha: str
    html_url: str
    download_url: str | None


class GitHubAPIError(RuntimeError):
    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"GitHub API request failed ({status_code}): {detail}")


class GitHubNotesClient:
    def __init__(self) -> None:
        self.settings = get_settings()

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {self.settings.github_token}",
            "User-Agent": USER_AGENT,
            "X-GitHub-Api-Version": "2022-11-28",
        }

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        url = f"{self.settings.github_api_url.rstrip('/')}{path}"
        response = httpx.request(
            method,
            url,
            headers=self._headers,
            timeout=20.0,
            **kwargs,
        )
        if response.status_code >= 400:
            detail = response.text
            try:
                payload = response.json()
                detail = payload.get("message", detail)
            except ValueError:
                pass
            raise GitHubAPIError(response.status_code, detail)
        if not response.content:
            return None
        return response.json()

    def _repo_html_url(self) -> str:
        return f"https://github.com/{self.settings.github_repo}"

    def _file_html_url(self, filename: str) -> str:
        return f"{self._repo_html_url()}/blob/{self.settings.github_branch}/{filename}"

    def list_notes(self) -> list[NoteSummary]:
        try:
            payload = self._request(
                "GET",
                f"/repos/{self.settings.github_repo}/contents",
                params={"ref": self.settings.github_branch},
            )
        except GitHubAPIError as exc:
            if exc.status_code == 409 and "empty" in exc.detail.lower():
                return []
            if exc.status_code == 404:
                return []
            raise
        if not isinstance(payload, list):
            raise RuntimeError("Expected a directory listing from GitHub.")

        notes = [
            NoteSummary(
                name=item["name"],
                path=item["path"],
                sha=item["sha"],
                html_url=item["html_url"],
                download_url=item.get("download_url"),
            )
            for item in payload
            if item.get("type") == "file" and item.get("name", "").endswith(".md")
        ]
        return sorted(notes, key=lambda note: note.name, reverse=True)

    def read_note(self, filename: str) -> tuple[str, str]:
        normalized_name = filename.strip().lstrip("/")
        payload = self._request(
            "GET",
            f"/repos/{self.settings.github_repo}/contents/{normalized_name}",
            params={"ref": self.settings.github_branch},
        )
        if payload.get("type") != "file":
            raise RuntimeError(f"{filename} is not a markdown file.")

        content = payload.get("content", "").replace("\n", "")
        decoded = base64.b64decode(content).decode("utf-8")
        return payload["path"], decoded

    def _existing_note_names(self) -> set[str]:
        return {note.name for note in self.list_notes()}

    def build_note_filename(self, summary: str) -> str:
        today = date.today().isoformat()
        base_name = f"{today}-{slugify_summary(summary)}"
        filename = f"{base_name}.md"
        existing = self._existing_note_names()
        if filename not in existing:
            return filename

        suffix = 2
        while f"{base_name}-{suffix}.md" in existing:
            suffix += 1
        return f"{base_name}-{suffix}.md"

    def write_note(self, filename: str, markdown: str, *, commit_message: str) -> dict[str, str]:
        payload = {
            "branch": self.settings.github_branch,
            "content": base64.b64encode(markdown.encode("utf-8")).decode("utf-8"),
            "message": commit_message,
        }
        result = self._request(
            "PUT",
            f"/repos/{self.settings.github_repo}/contents/{filename}",
            json=payload,
        )
        content = result.get("content", {})
        return {
            "filename": content.get("path", filename),
            "sha": content.get("sha", ""),
            "html_url": content.get("html_url", self._file_html_url(filename)),
        }
