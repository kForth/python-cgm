"""Raster extraction and rendering functions."""

from __future__ import annotations

__author__ = "Kestin Goforth"
__copyright__ = "Copyright 2026"
__license__ = "BSD-3-Clause"

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from cgm.extract.core import (
    coerce_int,
    extract_color_table,
    extract_color_value_extent,
    indexed_palette_bytes,
    parse_cell_array_metadata,
    render_raw_image_payload,
)
from cgm.extract.svg import extract_vector_svg_from_bytes
from cgm.extract.tiles import (
    _decode_tile_payload_to_image,
    _parse_text_tile_arrays,
    _parse_tile_arrays,
)
from cgm.parser import CELL_ARRAY_CLASS_ID, CELL_ARRAY_ELEMENT_ID, iter_elements
from cgm.types import RawImage

if TYPE_CHECKING:
    from cgm.types import CGMElement

log = logging.getLogger("cgm.extract")


def extract_raw_images_from_bytes(
    data: bytes,
    *,
    elements: list[CGMElement] | None = None,
) -> list[RawImage]:
    """Extract raw raster payloads from Cell Array elements in a CGM stream."""
    images: list[RawImage] = []
    image_index = 0
    parsed_elements = list(iter_elements(data)) if elements is None else elements

    if log.isEnabledFor(logging.DEBUG):
        log.debug("Starting extraction from %d bytes", len(data))

    for element in parsed_elements:
        if element.class_id != CELL_ARRAY_CLASS_ID or element.element_id != CELL_ARRAY_ELEMENT_ID:
            continue

        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                "Found Cell Array element: offset=%d param_length=%d",
                element.offset,
                len(element.parameters),
            )

        metadata = parse_cell_array_metadata(element.parameters)
        width = metadata["width"] if isinstance(metadata["width"], int) else None
        height = metadata["height"] if isinstance(metadata["height"], int) else None
        payload_offset = (
            metadata["payload_offset"] if isinstance(metadata["payload_offset"], int) else 0
        )
        payload = (
            element.parameters[payload_offset:] if payload_offset < len(element.parameters) else b""
        )

        images.append(
            RawImage(
                index=image_index,
                element_offset=element.offset,
                payload=payload,
                width=width,
                height=height,
                local_color_precision=metadata["local_color_precision"]
                if isinstance(metadata["local_color_precision"], int)
                else None,
                cell_representation_mode=metadata["cell_representation_mode"]
                if isinstance(metadata["cell_representation_mode"], int)
                else None,
            )
        )
        image_index += 1

    if log.isEnabledFor(logging.DEBUG):
        log.debug("Extraction complete: found %d Cell Array image(s)", len(images))

    return images


def extract_raw_images(file_path: str | Path) -> list[RawImage]:
    """Extract raw images from a CGM file path."""
    path = Path(file_path)
    raw = path.read_bytes()
    if log.isEnabledFor(logging.DEBUG):
        log.debug("Loaded CGM file %s (%d bytes)", path, len(raw))
    return extract_raw_images_from_bytes(raw)


def extract_raw_images_to_directory(
    file_path: str | Path,
    output_dir: str | Path,
    *,
    stem: str = "image",
) -> list[Path]:
    """Extract and write all raw payloads to an output directory."""
    written: list[Path] = []

    for image in extract_raw_images(file_path):
        path = image.write(output_dir, stem=stem)
        written.append(path)

    return written


def extract_rendered_images_to_directory(
    file_path: str | Path,
    output_dir: str | Path,
    *,
    stem: str = "image",
    debug_report: bool = False,
) -> list[Path]:
    """Write a composed SVG with raster background and vector overlays."""
    path = Path(file_path)
    raw = path.read_bytes()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    svg_path = out_dir / f"{stem}_0000.svg"
    svg_path.write_text(
        extract_vector_svg_from_bytes(raw),
        encoding="utf-8",
    )
    written.append(svg_path)

    if debug_report:
        arrays_report: list[dict[str, object]] = []
        for idx, array in enumerate(_parse_tile_arrays(raw)):
            tiles = array.get("tiles", [])
            arrays_report.append(
                {
                    "array_index": idx,
                    "cols": coerce_int(array.get("cols", 1)),
                    "rows": coerce_int(array.get("rows", 1)),
                    "tile_width": coerce_int(array.get("tile_width", 1)),
                    "tile_height": coerce_int(array.get("tile_height", 1)),
                    "total_width": coerce_int(array.get("total_width", 1)),
                    "total_height": coerce_int(array.get("total_height", 1)),
                    "tile_count": len(tiles) if isinstance(tiles, list) else 0,
                }
            )

        report = {
            "source": str(path),
            "arrays": arrays_report,
            "raw_image_count": len(extract_raw_images_from_bytes(raw)),
        }
        report_path = out_dir / f"{stem}_decode_report.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        written.append(report_path)

    return written


__all__ = [
    "_decode_tile_payload_to_image",
    "_parse_text_tile_arrays",
    "extract_color_table",
    "extract_color_value_extent",
    "extract_raw_images",
    "extract_raw_images_from_bytes",
    "extract_raw_images_to_directory",
    "extract_rendered_images_to_directory",
    "indexed_palette_bytes",
    "render_raw_image_payload",
]
