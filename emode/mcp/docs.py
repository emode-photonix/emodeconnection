"""Docs pages as Markdown: from a local directory if present, else the docs site."""

from __future__ import annotations

import os
import re
from collections.abc import Callable
from html.parser import HTMLParser
from pathlib import Path
from typing import ClassVar

DOCS_URL = "https://docs.emodephotonix.com/"
DOCS_DIR_ENV = "EMODE_DOCS_DIR"
_PAGE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_\-./]*")
_SUFFIXES = (".md", ".rst", ".txt")


class DocsError(Exception):
    """A docs page that cannot be read."""


def default_fetch(url: str) -> str:
    import requests

    response = requests.get(url, timeout=20)
    response.raise_for_status()
    return response.text


def valid_page(page: str) -> bool:
    return bool(_PAGE.fullmatch(page)) and ".." not in page.split("/")


def read_page(
    page: str, docs_dir: Path | str | None, fetch: Callable[[str], str] | None
) -> str:
    """The text of a docs page, such as ``examples/Ring_Design``, as Markdown.

    A page found under ``docs_dir`` (default: the directory named by the
    ``EMODE_DOCS_DIR`` environment variable) is returned as written. Otherwise
    the page is fetched from the docs site and its HTML converted.
    """
    if not valid_page(page):
        raise DocsError(f"{page!r} is not a docs page name")
    directory = docs_dir if docs_dir is not None else os.environ.get(DOCS_DIR_ENV)
    if directory:
        for suffix in _SUFFIXES:
            candidate = Path(directory) / f"{page}{suffix}"
            if candidate.is_file():
                return candidate.read_text()
    url = f"{DOCS_URL}{page}.html"
    try:
        html = (fetch or default_fetch)(url)
    except Exception as err:
        raise DocsError(
            f"Could not read the docs page {page!r} ({url}): {err}"
        ) from err
    return html_to_markdown(html)


class _Markdown(HTMLParser):
    _SKIP: ClassVar[frozenset[str]] = frozenset(
        {"script", "style", "head", "nav", "footer"}
    )

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.out: list[str] = []
        self._skip = 0
        self._pre = False

    def _block(self) -> None:
        if self.out and not "".join(self.out[-2:]).endswith("\n\n"):
            self.out.append("\n\n" if not self.out[-1].endswith("\n") else "\n")

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIP:
            self._skip += 1
        elif self._skip:
            return
        elif tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            self._block()
            self.out.append("#" * int(tag[1]) + " ")
        elif tag == "p":
            self._block()
        elif tag == "li":
            self.out.append("\n- ")
        elif tag == "pre":
            self._block()
            self.out.append("```\n")
            self._pre = True
        elif tag == "code" and not self._pre:
            self.out.append("`")
        elif tag == "br":
            self.out.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP:
            self._skip = max(0, self._skip - 1)
        elif self._skip:
            return
        elif tag in {"h1", "h2", "h3", "h4", "h5", "h6", "p"} or tag in {"ul", "ol"}:
            self.out.append("\n\n")
        elif tag == "pre":
            self.out.append("\n```\n\n")
            self._pre = False
        elif tag == "code" and not self._pre:
            self.out.append("`")

    def handle_data(self, data: str) -> None:
        if self._skip:
            return
        self.out.append(data if self._pre else re.sub(r"\s+", " ", data))


def html_to_markdown(html: str) -> str:
    """A plain Markdown rendering of an HTML page: headings, paragraphs, lists, code."""
    parser = _Markdown()
    parser.feed(html)
    text = "".join(parser.out)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return re.sub(r"\n\n(- )", r"\n\1", text).replace("\n\n- ", "\n- ") + "\n"
