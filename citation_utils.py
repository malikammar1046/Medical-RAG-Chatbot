import os
from typing import Any


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None

    text = str(value).strip()
    if not text or text.upper() in {"N/A", "UNKNOWN"}:
        return None

    return text


def _source_name(metadata: dict[str, Any]) -> str:
    title = _clean_text(metadata.get("title"))
    if title:
        return title

    source = _clean_text(metadata.get("source"))
    if source:
        return os.path.basename(source)

    return "Unknown"


def _page_number(metadata: dict[str, Any]) -> str:
    page_label = _clean_text(metadata.get("page_label"))
    if page_label:
        return page_label

    page = metadata.get("page")
    if isinstance(page, int):
        return str(page + 1)

    page_text = _clean_text(page)
    if page_text and page_text.isdigit():
        return str(int(page_text) + 1)

    return page_text or "N/A"


def format_document_references(documents: list[Any]) -> list[dict[str, str]]:
    references: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for doc in documents:
        metadata = getattr(doc, "metadata", {}) or {}
        source = _source_name(metadata)
        page = _page_number(metadata)
        key = (source, page)

        if key in seen:
            continue

        seen.add(key)
        references.append({"file": source, "page": page})

    return references


def format_references_markdown(references: list[dict[str, str]]) -> str:
    if not references:
        return ""

    lines = ["## References"]
    lines.extend(f"- {ref['file']}, Page {ref['page']}" for ref in references)
    return "\n".join(lines)
