"""CGM snapshot and JSON export helpers for parsed elements, payloads, hotspots, and SVG."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .hotspots import extract_hotspots_from_bytes
from .parser import iter_elements
from .raw_images import extract_raw_images_from_bytes
from .rendering import extract_vector_svg_from_bytes

log = logging.getLogger("cgm.snapshots")


def _build_data_snapshot(data: bytes) -> dict[str, object]:

    element_histogram: dict[tuple[int, int], int] = {}
    elements: list[dict[str, object]] = []
    element29_analysis: list[dict[str, object]] = []

    for element in iter_elements(data):
        key = (element.class_id, element.element_id)
        element_histogram[key] = element_histogram.get(key, 0) + 1
        elements.append(
            {
                "offset": element.offset,
                "class_id": element.class_id,
                "element_id": element.element_id,
                "parameter_length": len(element.parameters),
                "parameters_hex": element.parameters.hex(),
            }
        )
        if element.class_id == 4 and element.element_id == 29:
            element29_analysis.append(
                {
                    "offset": element.offset,
                    "length": len(element.parameters),
                    "ascii_ratio": 0.0,
                    "entropy_bits_per_byte": 0.0,
                    "top_bytes": [],
                    "ascii_runs": [],
                    "head_u16": [],
                    "restricted_text_detected": False,
                    "likely_binary_payload": False,
                }
            )

    raw_images = extract_raw_images_from_bytes(data)
    hotspots = extract_hotspots_from_bytes(data)
    raw_image_items = [
        {
            "index": image.index,
            "element_offset": image.element_offset,
            "width": image.width,
            "height": image.height,
            "payload_size": len(image.payload),
            "payload_hex": image.payload.hex(),
        }
        for image in raw_images
    ]

    hotspot_items = [
        {
            "index": hotspot.index,
            "source_tag": hotspot.source_tag,
            "name": hotspot.name,
            "x_min": hotspot.x_min,
            "y_min": hotspot.y_min,
            "x_max": hotspot.x_max,
            "y_max": hotspot.y_max,
            "raw_region_hex": hotspot.raw_region_hex,
        }
        for hotspot in hotspots
    ]

    histogram_items = [
        {
            "class_id": class_id,
            "element_id": element_id,
            "count": count,
        }
        for (class_id, element_id), count in sorted(
            element_histogram.items(), key=lambda item: item[1], reverse=True
        )
    ]

    return {
        "byte_length": len(data),
        "element_count": len(elements),
        "element_histogram": histogram_items,
        "elements": elements,
        "element29_analysis": element29_analysis,
        "raw_images": raw_image_items,
        "hotspots": hotspot_items,
        "vector_svg": extract_vector_svg_from_bytes(data),
    }


def extract_data_json_from_bytes(data: bytes) -> str:
    return json.dumps(_build_data_snapshot(data), indent=2)


def extract_data_json(file_path: str | Path) -> str:
    raw = Path(file_path).read_bytes()
    if log.isEnabledFor(logging.DEBUG):
        log.debug("Loaded CGM file %s (%d bytes) for JSON export", file_path, len(raw))
    return extract_data_json_from_bytes(raw)


def extract_data_json_to_directory(
    file_path: str | Path,
    output_dir: str | Path,
    *,
    stem: str = "image",
) -> Path:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{stem}_0000.json"
    json_text = extract_data_json(file_path)
    target.write_text(json_text + "\n", encoding="utf-8")
    if log.isEnabledFor(logging.DEBUG):
        log.debug("Wrote data JSON: %s", target)
    return target
