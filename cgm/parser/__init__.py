"""Low-level parser for binary and clear-text CGM files (ISO/IEC 8632)."""

from __future__ import annotations

__author__ = "Kestin Goforth"
__copyright__ = "Copyright 2026"
__license__ = "BSD-3-Clause"

from typing import TYPE_CHECKING

from cgm.parser.binary import iter_binary_elements
from cgm.parser.constants import CELL_ARRAY_CLASS_ID, CELL_ARRAY_ELEMENT_ID, LONG_FORM_SENTINEL
from cgm.parser.text import iter_text_elements

if TYPE_CHECKING:
    from collections.abc import Iterator

    from cgm.types import CGMElement


def _looks_like_text_cgm(data: bytes) -> bool:
    if not data:
        return False

    head = data[:512]
    if b"\x00" in head:
        return False

    control_bytes = sum(byte < 9 or (13 < byte < 32) for byte in head)
    if control_bytes > max(4, len(head) // 20):
        return False

    try:
        sample = head.decode("ascii")
    except UnicodeDecodeError:
        return False

    upper = sample.upper()
    if ";" not in upper:
        return False

    keywords = (
        "BEGMF",
        "LINE",
        "POLYGON",
        "TEXT",
        "TRANSPARENCY",
        "TRANSPARENCYMODE",
        "CLIPRECT",
        "CLIPIND",
        "COLRVALUEEXT",
        "COLORVALUEEXTENT",
        "CELLARRAY",
        "BEGTILEARRAY",
        "BITONALTILE",
        "COLORTILE",
        "COLOURTILE",
        "DIRECTCOLORTILE",
        "DIRECTCOLOURTILE",
        "INDEXCOLORTILE",
        "INDEXCOLOURTILE",
        "MONOCHROMETILE",
        "APD",
        "BEGAPS",
        "ENDAPS",
    )
    return any(token in upper for token in keywords)


def iter_elements(data: bytes) -> Iterator[CGMElement]:
    """Yield parsed elements from either binary or clear-text CGM streams."""
    if _looks_like_text_cgm(data):
        yield from iter_text_elements(data)
        return

    yield from iter_binary_elements(data)


__all__ = [
    "CELL_ARRAY_CLASS_ID",
    "CELL_ARRAY_ELEMENT_ID",
    "LONG_FORM_SENTINEL",
    "iter_elements",
]
