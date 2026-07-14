"""Core data structures for parsed CGM content."""

from __future__ import annotations

__author__ = "Kestin Goforth"
__copyright__ = "Copyright 2026"
__license__ = "BSD-3-Clause"

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True, frozen=True)
class CGMElement:
    """Represents one CGM element as parsed from a binary CGM stream."""

    class_id: int
    element_id: int
    parameters: bytes
    offset: int


@dataclass(slots=True, frozen=True)
class RawImage:
    """Represents extracted raw raster bytes from a CGM Cell Array element."""

    index: int
    element_offset: int
    payload: bytes
    width: int | None = None
    height: int | None = None

    def default_filename(self, *, stem: str = "image") -> str:
        """Return a deterministic filename for saving this payload."""
        return f"{stem}_{self.index:04d}.bin"

    def write(self, output_dir: str | Path, *, stem: str = "image") -> Path:
        """Write this payload to disk and return the resulting path."""
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / self.default_filename(stem=stem)
        target.write_bytes(self.payload)
        return target


@dataclass(slots=True, frozen=True)
class HotSpot:
    """Represents a hotspot region recovered from APD metadata or APS geometry."""

    index: int
    source_tag: str | None
    name: str | None
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    raw_region_hex: str
