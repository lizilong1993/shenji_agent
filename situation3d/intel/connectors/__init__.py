"""External VGI connectors (P0).

This package is intentionally independent of ``sim``/``engine``: it fetches,
archives and normalizes observations; the gateway/scheduler decide what a side
may consume.
"""

from .archive import ArchiveRecord, archive_raw, read_archive
from .fetch import FetchResult, SourceStatus, build_bundle
from .http import FetchError, MissingCredential, fetch_bytes, fetch_json, fetch_text
from .registry import SOURCES, SOURCE_IDS, SourceSpec, get_source

__all__ = [
    "ArchiveRecord",
    "FetchError",
    "FetchResult",
    "MissingCredential",
    "SOURCES",
    "SOURCE_IDS",
    "SourceSpec",
    "SourceStatus",
    "archive_raw",
    "build_bundle",
    "fetch_bytes",
    "fetch_json",
    "fetch_text",
    "get_source",
    "read_archive",
]
