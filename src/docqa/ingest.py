"""Load reStructuredText docs and split them into section-aware chunks."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

HEADING_CHARS = set("=-~^\"'`*+#:._")
DIRECTIVE = re.compile(r"^\s*\.\. [\w:-]+::.*$")
ROLE = re.compile(r":(?:class|func|meth|ref|term|math|attr|mod|doc|data|obj|paramref|user|issue|pr|wiki|kbd):`(?:[^`<]*<)?([^`>]+)>?`")


@dataclass
class Chunk:
    chunk_id: str
    source: str          # file name, e.g. "cross_validation.rst"
    section: str         # nearest heading above the chunk
    text: str
    meta: dict = field(default_factory=dict)

    @property
    def citation(self) -> str:
        return f"{self.source} › {self.section}"


def _is_underline(line: str, title: str) -> bool:
    s = line.strip()
    return (len(s) >= 3 and len(set(s)) == 1 and s[0] in HEADING_CHARS
            and len(s) >= len(title.strip()) - 2 and title.strip() != "")


def clean_rst(text: str) -> str:
    """Strip the most common reST markup so the text reads as plain prose."""
    text = ROLE.sub(lambda m: m.group(1).split(".")[-1], text)
    text = re.sub(r"``([^`]+)``", r"\1", text)
    text = re.sub(r"`([^`<]+)\s*<[^>]+>`_+", r"\1", text)
    text = re.sub(r"\|([\w-]+)\|", r"\1", text)
    text = re.sub(r"\[[#*A-Za-z0-9_-]*\]_", "", text)
    text = re.sub(r"\.\. \|[^|]+\|\s*\w+::\s*\S*", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def parse_sections(path: Path) -> list[tuple[str, str]]:
    """Return (heading, body) pairs for one .rst file."""
    lines = path.read_text(encoding="utf-8").splitlines()
    sections, title, buf = [], path.stem.replace("_", " ").title(), []
    i = 0
    while i < len(lines):
        line = lines[i]
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if line.strip() and not line.startswith(" ") and _is_underline(nxt, line):
            if buf:
                sections.append((title, "\n".join(buf)))
            title, buf = line.strip(), []
            i += 2
            continue
        if _is_underline(line, nxt) and nxt.strip() and i + 2 < len(lines) and _is_underline(lines[i + 2], nxt):
            if buf:  # overlined heading
                sections.append((title, "\n".join(buf)))
            title, buf = nxt.strip(), []
            i += 3
            continue
        buf.append(line)
        i += 1
    if buf:
        sections.append((title, "\n".join(buf)))
    return sections


def _drop_directive_noise(body: str) -> str:
    out, skip_indent = [], None
    for line in body.splitlines():
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())
        if skip_indent is not None:
            if stripped == "" or indent > skip_indent:
                continue
            skip_indent = None
        if re.match(r"^\.\. \|[^|]+\|", stripped):
            skip_indent = indent
            continue
        if re.match(r"^\.\. (image|figure|currentmodule|_[\w-]+|raw|only|include|plot)::?", stripped) or stripped.startswith(".. _"):
            skip_indent = indent if re.match(r"^\.\. (image|figure|raw|only|plot)::", stripped) else None
            continue
        if re.match(r"^:(target|scale|align|width|height|alt|class|name|color|open|caption):", stripped):
            continue
        if stripped.startswith(">>>") or stripped.startswith("..."):
            continue  # doctest code
        out.append(line)
    return "\n".join(out)


def chunk_docs(doc_dir: str | Path, max_words: int = 180, overlap: int = 30) -> list[Chunk]:
    """Split every section into overlapping word windows, keeping the heading with each chunk."""
    chunks: list[Chunk] = []
    for path in sorted(Path(doc_dir).glob("*.rst")):
        for s_idx, (heading, body) in enumerate(parse_sections(path)):
            text = clean_rst(_drop_directive_noise(body))
            words = text.split()
            if len(words) < 25:
                continue
            step = max_words - overlap
            for w_idx, start in enumerate(range(0, max(len(words) - overlap, 1), step)):
                piece = " ".join(words[start:start + max_words])
                chunks.append(Chunk(f"{path.stem}:{s_idx}:{w_idx}", path.name, clean_rst(heading), piece))
    return chunks
