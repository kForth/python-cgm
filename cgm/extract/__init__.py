"""High-level CGM extraction package grouped by functionality."""

from __future__ import annotations

__author__ = "Kestin Goforth"
__copyright__ = "Copyright 2026"
__license__ = "BSD-3-Clause"

from cgm.extract.data_json import (
    extract_data_json,
    extract_data_json_from_bytes,
    extract_data_json_to_directory,
)
from cgm.extract.hotspots import (
    extract_final_image_and_hotspots,
    extract_hotspots,
    extract_hotspots_from_bytes,
    extract_hotspots_to_directory,
)
from cgm.extract.raster import (
    _decode_tile_payload_to_image,
    _parse_text_tile_arrays,
    extract_color_table,
    extract_color_value_extent,
    extract_raw_images,
    extract_raw_images_from_bytes,
    extract_raw_images_to_directory,
    extract_rendered_images_to_directory,
    indexed_palette_bytes,
    render_raw_image_payload,
)
from cgm.extract.svg import (
    extract_vector_svg,
    extract_vector_svg_from_bytes,
    extract_vector_svg_to_directory,
)
from cgm.types import HotSpot, RawImage

__all__ = [
    "HotSpot",
    "RawImage",
    "_decode_tile_payload_to_image",
    "_parse_text_tile_arrays",
    "extract_color_table",
    "extract_color_value_extent",
    "extract_data_json",
    "extract_data_json_from_bytes",
    "extract_data_json_to_directory",
    "extract_final_image_and_hotspots",
    "extract_hotspots",
    "extract_hotspots_from_bytes",
    "extract_hotspots_to_directory",
    "extract_raw_images",
    "extract_raw_images_from_bytes",
    "extract_raw_images_to_directory",
    "extract_rendered_images_to_directory",
    "extract_vector_svg",
    "extract_vector_svg_from_bytes",
    "extract_vector_svg_to_directory",
    "indexed_palette_bytes",
    "render_raw_image_payload",
]
