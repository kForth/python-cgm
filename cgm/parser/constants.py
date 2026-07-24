"""Shared parser constants."""

from __future__ import annotations

__author__ = "Kestin Goforth"
__copyright__ = "Copyright 2026"
__license__ = "BSD-3-Clause"

LONG_FORM_SENTINEL = 0x1F

# Class 4, Element 9 is Cell Array (raster data carrier in CGM).
CELL_ARRAY_CLASS_ID = 4
CELL_ARRAY_ELEMENT_ID = 9

_TEXT_TILE_COMMANDS = {
    "BITONALTILE",
    "COLORTILE",
    "COLOURTILE",
    "DIRECTCOLORTILE",
    "DIRECTCOLOURTILE",
    "INDEXCOLORTILE",
    "INDEXCOLOURTILE",
    "MONOCHROMETILE",
}
