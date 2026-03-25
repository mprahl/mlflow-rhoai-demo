"""Tooling for note taking, web search, and GitHub-backed note storage."""

from __future__ import annotations

import re
from datetime import date
from textwrap import dedent
from typing import Any
from urllib.parse import quote

import httpx
from ddgs import DDGS
from langchain_core.tools import ToolException, tool

from mlflow_notes_agent.github_notes import GitHubNotesClient

USER_AGENT = "mlflow-notes-demo/0.1"


def _strip_html(text: str) -> str:
    return re.sub(r"<[^>]+>", "", text).strip()


def _format_search_results(title: str, results: list[dict[str, Any]]) -> str:
    if not results:
        return f"No {title.lower()} results found."

    lines = [f"{title} results:"]
    for index, result in enumerate(results, start=1):
        description = result.get("description") or result.get("body") or result.get("snippet") or ""
        lines.extend(
            [
                f"{index}. {result.get('title', 'Untitled')}",
                f"   URL: {result.get('url', '')}",
                f"   Summary: {description}",
            ]
        )
    return "\n".join(lines)


@tool
def search_duckduckgo(query: str, max_results: int = 5) -> str:
    """Search the web with DuckDuckGo when you need current external context."""
    if max_results < 1 or max_results > 10:
        raise ValueError("max_results must be between 1 and 10.")

    with DDGS(timeout=20) as ddgs:
        items = list(ddgs.text(query, max_results=max_results))

    results = [
        {
            "title": item.get("title", ""),
            "url": item.get("href", ""),
            "description": item.get("body", ""),
        }
        for item in items
    ]
    return _format_search_results("DuckDuckGo", results)


@tool
def search_wikipedia(query: str, max_results: int = 3) -> str:
    """Search Wikipedia for reference material and concise encyclopedic context."""
    if max_results < 1 or max_results > 10:
        raise ValueError("max_results must be between 1 and 10.")

    response = httpx.get(
        "https://en.wikipedia.org/w/rest.php/v1/search/page",
        params={"q": query, "limit": max_results},
        headers={"User-Agent": USER_AGENT},
        timeout=20.0,
    )
    response.raise_for_status()
    payload = response.json()

    results = []
    for page in payload.get("pages", []):
        title = page.get("title", "")
        page_key = page.get("key", title.replace(" ", "_"))
        results.append(
            {
                "title": title,
                "url": f"https://en.wikipedia.org/wiki/{quote(page_key, safe=':_()')}",
                "description": " ".join(
                    part
                    for part in [
                        page.get("description", ""),
                        _strip_html(page.get("excerpt", "")),
                    ]
                    if part
                ),
            }
        )

    return _format_search_results("Wikipedia", results)


@tool
def list_notes(limit: int = 20) -> str:
    """List existing markdown notes stored in the configured GitHub repository branch."""
    if limit < 1 or limit > 100:
        raise ValueError("limit must be between 1 and 100.")

    try:
        client = GitHubNotesClient()
        notes = client.list_notes()[:limit]
    except RuntimeError as exc:
        raise ToolException(str(exc)) from exc

    if not notes:
        return "No markdown notes were found in the configured GitHub branch."

    lines = ["Available notes:"]
    for index, note in enumerate(notes, start=1):
        lines.append(f"{index}. {note.name} ({note.html_url})")
    return "\n".join(lines)


@tool
def read_note(filename: str) -> str:
    """Read a previously saved markdown note by filename."""
    try:
        client = GitHubNotesClient()
        path, content = client.read_note(filename)
    except RuntimeError as exc:
        raise ToolException(str(exc)) from exc

    return f"Note: {path}\n\n{content}"


@tool
def save_note(
    title: str,
    summary: str,
    notes_markdown: str,
    source_urls: list[str] | None = None,
) -> str:
    """Save a markdown note to GitHub. Use only when the user asks to create or store a note."""
    try:
        client = GitHubNotesClient()
        filename = client.build_note_filename(summary)

        source_lines = ""
        cleaned_sources = [source.strip() for source in (source_urls or []) if source.strip()]
        if cleaned_sources:
            source_lines = "\n## Sources\n" + "\n".join(f"- {source}" for source in cleaned_sources)

        markdown = dedent(
            f"""\
            # {title}

            - Date: {date.today().isoformat()}
            - Summary: {summary.strip()}

            ## Notes
            {notes_markdown.strip()}
            {source_lines}
            """
        ).strip() + "\n"

        result = client.write_note(
            filename,
            markdown,
            commit_message=f"Add note {filename}",
        )
    except RuntimeError as exc:
        raise ToolException(str(exc)) from exc

    return (
        f"Saved note to {result['filename']} in GitHub. "
        f"View it at {result['html_url'] or 'the configured repository branch'}."
    )


NOTES_TOOLS = [
    search_duckduckgo,
    search_wikipedia,
    list_notes,
    read_note,
    save_note,
]
