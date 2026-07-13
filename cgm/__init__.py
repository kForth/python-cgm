"""Python tools for extracting CGM payloads and exporting best-effort SVG/JSON."""

__author__ = "Kestin Goforth"
__copyright__ = "Copyright 2026"
__license__ = "BSD-3-Clause"
__version__ = "0.1.0"

import logging

from .errors import CGMError, CGMParseError
from .extract import (
    extract_data_json,
    extract_data_json_from_bytes,
    extract_data_json_to_directory,
    extract_raw_images,
    extract_raw_images_from_bytes,
    extract_raw_images_to_directory,
    extract_vector_svg,
    extract_vector_svg_from_bytes,
    extract_vector_svg_to_directory,
)
from .types import CGMElement, RawImage

log = logging.getLogger("cgm")

__all__ = [
    "CGMElement",
    "CGMError",
    "CGMParseError",
    "RawImage",
    "extract_data_json",
    "extract_data_json_from_bytes",
    "extract_data_json_to_directory",
    "extract_raw_images",
    "extract_raw_images_from_bytes",
    "extract_raw_images_to_directory",
    "extract_vector_svg",
    "extract_vector_svg_from_bytes",
    "extract_vector_svg_to_directory",
]
