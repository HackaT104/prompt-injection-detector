"""Extract and chunk untrusted external content for indirect-injection analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from html.parser import HTMLParser
from io import BytesIO
from pathlib import Path
from typing import Any
import re
import zipfile
from xml.etree import ElementTree

from src.preprocessing import clean_text


SUPPORTED_SOURCE_TYPES = {"raw_text", "txt", "pdf", "docx", "html", "web_html"}
TEXT_EXTENSIONS = {".txt", ".md", ".csv", ".json", ".jsonl"}


@dataclass(frozen=True)
class SourceMetadata:
    source_type: str
    source_name: str
    trust_level: str = "untrusted"


@dataclass(frozen=True)
class ExtractedSegment:
    text: str
    page_number: int | None = None


@dataclass(frozen=True)
class ExternalContentChunk:
    text: str
    cleaned_text: str
    source_type: str
    source_name: str
    trust_level: str
    chunk_id: str
    page_number: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class _VisibleHTMLParser(HTMLParser):
    BLOCK_TAGS = {"p", "div", "br", "li", "tr", "td", "th", "h1", "h2", "h3", "h4", "section", "article"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        lowered = tag.lower()
        if lowered in {"script", "style", "noscript"}:
            self._ignored_depth += 1
        elif lowered in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        lowered = tag.lower()
        if lowered in {"script", "style", "noscript"} and self._ignored_depth:
            self._ignored_depth -= 1
        elif lowered in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth:
            self.parts.append(data)

    def text(self) -> str:
        return re.sub(r"\n{3,}", "\n\n", "".join(self.parts)).strip()


def normalize_source_type(source_type: str, source_name: str = "") -> str:
    value = str(source_type or "auto").strip().lower().replace("-", "_")
    aliases = {"text": "raw_text", "raw": "raw_text", "web": "web_html", "htm": "html"}
    value = aliases.get(value, value)
    if value == "auto":
        suffix = Path(source_name).suffix.lower()
        if suffix in TEXT_EXTENSIONS:
            return "txt"
        if suffix in {".html", ".htm"}:
            return "html"
        if suffix == ".pdf":
            return "pdf"
        if suffix == ".docx":
            return "docx"
        return "raw_text"
    if value not in SUPPORTED_SOURCE_TYPES:
        raise ValueError(f"Unsupported source_type '{source_type}'. Supported: {sorted(SUPPORTED_SOURCE_TYPES)}")
    return value


def _decode_text(data: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _extract_pdf(data: bytes) -> list[ExtractedSegment]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise RuntimeError("PDF extraction requires 'pypdf'. Run: pip install pypdf") from exc

    reader = PdfReader(BytesIO(data))
    segments = [
        ExtractedSegment(text=(page.extract_text() or "").strip(), page_number=index)
        for index, page in enumerate(reader.pages, start=1)
    ]
    segments = [segment for segment in segments if segment.text]
    if not segments:
        raise ValueError("PDF contains no extractable text. Scanned-image OCR is not implemented yet.")
    return segments


def _extract_docx(data: bytes) -> list[ExtractedSegment]:
    try:
        with zipfile.ZipFile(BytesIO(data)) as archive:
            document_xml = archive.read("word/document.xml")
    except (zipfile.BadZipFile, KeyError) as exc:
        raise ValueError("Invalid DOCX file or missing word/document.xml.") from exc

    root = ElementTree.fromstring(document_xml)
    namespace = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs: list[str] = []
    for paragraph in root.findall(".//w:p", namespace):
        text = "".join(node.text or "" for node in paragraph.findall(".//w:t", namespace)).strip()
        if text:
            paragraphs.append(text)
    if not paragraphs:
        raise ValueError("DOCX contains no extractable text.")
    return [ExtractedSegment(text="\n".join(paragraphs))]


def _extract_html(text: str) -> str:
    parser = _VisibleHTMLParser()
    parser.feed(text)
    parser.close()
    return parser.text()


def extract_external_content(
    *,
    source_type: str,
    source_name: str,
    raw_text: str | None = None,
    content_bytes: bytes | None = None,
) -> tuple[SourceMetadata, list[ExtractedSegment]]:
    """Extract text without fetching remote URLs; all sources are forced to untrusted."""
    normalized_type = normalize_source_type(source_type, source_name)
    metadata = SourceMetadata(source_type=normalized_type, source_name=source_name or "inline-content")

    if normalized_type == "pdf":
        if not content_bytes:
            raise ValueError("PDF input requires binary content via content_bytes/upload endpoint.")
        segments = _extract_pdf(content_bytes)
    elif normalized_type == "docx":
        if not content_bytes:
            raise ValueError("DOCX input requires binary content via content_bytes/upload endpoint.")
        segments = _extract_docx(content_bytes)
    else:
        text = raw_text if raw_text is not None else _decode_text(content_bytes or b"")
        if normalized_type in {"html", "web_html"}:
            text = _extract_html(text)
        segments = [ExtractedSegment(text=text.strip())] if text.strip() else []

    if not segments:
        raise ValueError("External content is empty after text extraction.")
    return metadata, segments


def chunk_text(text: str, max_chars: int = 1200, overlap_chars: int = 160) -> list[str]:
    if max_chars < 200:
        raise ValueError("max_chars must be at least 200.")
    overlap_chars = max(0, min(int(overlap_chars), max_chars // 2))
    normalized = re.sub(r"[ \t]+", " ", str(text)).strip()
    if len(normalized) <= max_chars:
        return [normalized] if normalized else []

    chunks: list[str] = []
    start = 0
    while start < len(normalized):
        target = min(len(normalized), start + max_chars)
        end = target
        if target < len(normalized):
            candidates = [normalized.rfind(marker, start + max_chars // 2, target) for marker in ("\n\n", ". ", "! ", "? ", " ")]
            best = max(candidates)
            if best > start:
                end = best + 1
        chunk = normalized[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(normalized):
            break
        next_start = max(start + 1, end - overlap_chars)
        start = next_start
    return chunks


def build_external_chunks(
    metadata: SourceMetadata,
    segments: list[ExtractedSegment],
    *,
    max_chars: int = 1200,
    overlap_chars: int = 160,
) -> list[ExternalContentChunk]:
    chunks: list[ExternalContentChunk] = []
    counter = 1
    for segment in segments:
        for text in chunk_text(segment.text, max_chars=max_chars, overlap_chars=overlap_chars):
            chunks.append(
                ExternalContentChunk(
                    text=text,
                    cleaned_text=clean_text(text),
                    source_type=metadata.source_type,
                    source_name=metadata.source_name,
                    trust_level="untrusted",
                    chunk_id=f"chunk-{counter:04d}",
                    page_number=segment.page_number,
                )
            )
            counter += 1
    return chunks
