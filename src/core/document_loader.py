"""Extracts document text using docling, which preserves structure
(headings, tables) as markdown instead of flattening everything to plain text."""

from pathlib import Path

from docling.document_converter import DocumentConverter


def load_document_text(path: Path) -> str:
    converter = DocumentConverter()
    result = converter.convert(str(path))
    return result.document.export_to_markdown()
