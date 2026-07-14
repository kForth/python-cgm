"""Python tools for CGM parsing, final SVG composition, and hotspot extraction."""

__author__ = "Kestin Goforth"
__copyright__ = "Copyright 2026"
__license__ = "BSD-3-Clause"
__version__ = "0.2.0"

import logging

from .errors import CGMError, CGMParseError
from .extract import (
    extract_data_json,
    extract_data_json_from_bytes,
    extract_data_json_to_directory,
    extract_final_image_and_hotspots,
    extract_hotspots,
    extract_hotspots_from_bytes,
    extract_hotspots_to_directory,
    extract_raw_images,
    extract_raw_images_from_bytes,
    extract_raw_images_to_directory,
    extract_rendered_images_to_directory,
    extract_vector_svg,
    extract_vector_svg_from_bytes,
    extract_vector_svg_to_directory,
)
from .types import CGMElement, HotSpot, RawImage

log = logging.getLogger("cgm")

__all__ = [
    "CGMElement",
    "CGMError",
    "CGMParseError",
    "HotSpot",
    "RawImage",
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
]
