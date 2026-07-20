"""JSON export functions for parsed CGM content."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from cgm.extract.core import analyze_element29_payload
from cgm.extract.hotspots import extract_hotspots_from_bytes
from cgm.extract.raster import extract_raw_images_from_bytes
from cgm.extract.svg import extract_vector_svg_from_bytes
from cgm.parser import iter_elements


def _build_data_snapshot(
    data: bytes,
) -> dict[str, object]:
    element_histogram: Counter[tuple[int, int]] = Counter()
    elements: list[dict[str, object]] = []
    element29_analysis: list[dict[str, object]] = []

    for element in iter_elements(data):
        key = (element.class_id, element.element_id)
        element_histogram[key] += 1
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
                    **analyze_element29_payload(element.parameters),
                    "decode_candidates": [],
                    "has_plausible_decode": False,
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
            "local_color_precision": image.local_color_precision,
            "cell_representation_mode": image.cell_representation_mode,
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
        for (class_id, element_id), count in element_histogram.most_common()
    ]

    return {
        "byte_length": len(data),
        "element_count": len(elements),
        "element_histogram": histogram_items,
        "elements": elements,
        "element29_analysis": element29_analysis,
        "raw_images": raw_image_items,
        "hotspots": hotspot_items,
        "vector_svg": extract_vector_svg_from_bytes(
            data,
        ),
    }


def extract_data_json_from_bytes(
    data: bytes,
) -> str:
    """Serialize parsed CGM content, extracted payloads, and SVG into JSON."""
    snapshot = _build_data_snapshot(
        data,
    )
    return json.dumps(snapshot, indent=2)


def extract_data_json(
    file_path: str | Path,
) -> str:
    """Load a CGM file and serialize parsed data to JSON."""
    path = Path(file_path)
    raw = path.read_bytes()
    return extract_data_json_from_bytes(
        raw,
    )


def extract_data_json_to_directory(
    file_path: str | Path,
    output_dir: str | Path,
    *,
    stem: str = "image",
) -> Path:
    """Export parsed CGM data and metadata to a JSON file in an output directory."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{stem}_0000.json"
    json_text = extract_data_json(
        file_path,
    )
    target.write_text(json_text + "\n", encoding="utf-8")
    return target


__all__ = [
    "extract_data_json",
    "extract_data_json_from_bytes",
    "extract_data_json_to_directory",
]
