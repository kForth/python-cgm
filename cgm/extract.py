"""Public CGM extraction facade for SVG, hotspot, image, and JSON helpers."""

from __future__ import annotations

__author__ = "Kestin Goforth"
__copyright__ = "Copyright 2026"
__license__ = "BSD-3-Clause"

import logging
from pathlib import Path

from . import hotspots as _hotspots
from . import raw_images as _raw_images
from . import rendering as _rendering
from . import snapshots as _snapshots

log = logging.getLogger("cgm.extract")

extract_hotspots = _hotspots.extract_hotspots
extract_hotspots_from_bytes = _hotspots.extract_hotspots_from_bytes
extract_hotspots_to_directory = _hotspots.extract_hotspots_to_directory
extract_raw_images = _raw_images.extract_raw_images
extract_raw_images_from_bytes = _raw_images.extract_raw_images_from_bytes
extract_raw_images_to_directory = _raw_images.extract_raw_images_to_directory
extract_data_json = _snapshots.extract_data_json
extract_data_json_from_bytes = _snapshots.extract_data_json_from_bytes
extract_data_json_to_directory = _snapshots.extract_data_json_to_directory
extract_vector_svg_from_bytes = _rendering.extract_vector_svg_from_bytes
extract_vector_svg = _rendering.extract_vector_svg
extract_vector_svg_to_directory = _rendering.extract_vector_svg_to_directory
extract_rendered_images_to_directory = _rendering.extract_rendered_images_to_directory


def _score_bitmap(bits: list[int], width: int, height: int) -> float:
    return float(_rendering.score_bitmap(bits, width, height))


def extract_final_image_and_hotspots(file_path: str | Path) -> dict[str, object]:
    path = Path(file_path)
    raw = path.read_bytes()
    return {
        "image": _rendering.extract_vector_svg_from_bytes(raw),
        "hotspots": [
            {
                "index": item.index,
                "source_tag": item.source_tag,
                "name": item.name,
                "x_min": item.x_min,
                "y_min": item.y_min,
                "x_max": item.x_max,
                "y_max": item.y_max,
                "raw_region_hex": item.raw_region_hex,
            }
            for item in _hotspots.extract_hotspots_from_bytes(raw)
        ],
    }
