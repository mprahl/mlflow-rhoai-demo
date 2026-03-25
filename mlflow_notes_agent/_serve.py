"""Convenience wrapper to start the LangGraph API dev server."""

from __future__ import annotations

import subprocess
import sys


def main() -> None:
    subprocess.run(
        ["langgraph", "dev", "--no-browser", *sys.argv[1:]],
        check=False,
    )
