"""Hotspot extraction functions."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from cgm.extract.core import (
    decode_application_property,
    decode_hotspot_region_bbox,
    decode_point_pairs_exact,
    decode_prefixed_ascii,
    decode_restricted_text,
    parse_f32_be,
)
from cgm.extract.svg import extract_vector_svg
from cgm.parser import iter_elements
from cgm.types import HotSpot

log = logging.getLogger("cgm.extract")


def extract_hotspots_from_bytes(data: bytes) -> list[HotSpot]:
    """Extract hotspot regions from APD application data and APS geometry elements."""
    hotspots: list[HotSpot] = []

    context_stack: list[dict[str, object]] = []

    def _new_context(tag: str | None) -> dict[str, object]:
        return {
            "tag": tag,
            "name": None,
            "bbox": None,
            "region_hex": None,
            "geom_bbox": None,
        }

    def _update_geom_bbox(context: dict[str, object], x: float, y: float) -> None:
        existing = context.get("geom_bbox")
        if isinstance(existing, tuple) and len(existing) == 4:
            x_min, y_min, x_max, y_max = existing
            context["geom_bbox"] = (min(x_min, x), min(y_min, y), max(x_max, x), max(y_max, y))
            return
        context["geom_bbox"] = (x, y, x, y)

    def _bbox_from_geometry(context: dict[str, object]) -> tuple[int, int, int, int] | None:
        geom = context.get("geom_bbox")
        if not (isinstance(geom, tuple) and len(geom) == 4):
            return None
        gx_min, gy_min, gx_max, gy_max = geom
        x_min = round(float(gx_min))
        y_min = round(float(gy_min))
        x_max = round(float(gx_max))
        y_max = round(float(gy_max))
        if x_min >= x_max or y_min >= y_max:
            return None
        return x_min, y_min, x_max, y_max

    def _flush_context(context: dict[str, object]) -> None:
        bbox = context.get("bbox")
        if not (isinstance(bbox, tuple) and len(bbox) == 4):
            bbox = _bbox_from_geometry(context)
        if not (isinstance(bbox, tuple) and len(bbox) == 4):
            return

        x_min, y_min, x_max, y_max = bbox
        name_value = context.get("name")
        tag_value = context.get("tag")
        name: str | None
        if isinstance(name_value, str) and name_value:
            name = name_value
        elif isinstance(tag_value, str) and tag_value:
            name = tag_value
        else:
            name = None

        source_tag = tag_value if isinstance(tag_value, str) else None
        raw_region_hex = context.get("region_hex")
        region_hex = raw_region_hex if isinstance(raw_region_hex, str) else ""

        hotspots.append(
            HotSpot(
                index=len(hotspots),
                source_tag=source_tag,
                name=name,
                x_min=x_min,
                y_min=y_min,
                x_max=x_max,
                y_max=y_max,
                raw_region_hex=region_hex,
            )
        )

    for element in iter_elements(data):
        if element.class_id == 0 and element.element_id == 21:
            context_stack.append(_new_context(decode_prefixed_ascii(element.parameters)))
            continue

        if element.class_id == 9 and element.element_id == 1:
            if not context_stack:
                continue
            prop = decode_application_property(element.parameters)
            if prop is None:
                continue
            key, value = prop
            context = context_stack[-1]
            if key == "name":
                decoded = value.decode("ascii", errors="ignore").rstrip("\x00")
                context["name"] = decoded or None
            elif key == "region":
                bbox = decode_hotspot_region_bbox(value)
                if bbox is not None:
                    context["bbox"] = bbox
                    context["region_hex"] = value.hex()
            continue

        if element.class_id == 4 and context_stack:
            context = context_stack[-1]
            if element.element_id in (1, 7):
                for x, y in decode_point_pairs_exact(element.parameters):
                    _update_geom_bbox(context, x, y)
            elif element.element_id == 5 and len(element.parameters) >= 8:
                x_val = parse_f32_be(element.parameters[0:4])
                y_val = parse_f32_be(element.parameters[4:8])
                if x_val is not None and y_val is not None:
                    _update_geom_bbox(context, x_val, y_val)
            elif element.element_id == 29:
                restricted = decode_restricted_text(element.parameters)
                if restricted is not None:
                    anchor_x, anchor_y, box_w, box_h, _text = restricted
                    _update_geom_bbox(context, anchor_x, anchor_y)
                    _update_geom_bbox(context, anchor_x + box_w, anchor_y + box_h)
            continue

        if element.class_id == 0 and element.element_id in (22, 23):
            if context_stack:
                _flush_context(context_stack.pop())
            continue

    while context_stack:
        _flush_context(context_stack.pop())

    return hotspots


def extract_hotspots(file_path: str | Path) -> list[HotSpot]:
    """Extract hotspot regions from a CGM file path."""
    path = Path(file_path)
    raw = path.read_bytes()
    if log.isEnabledFor(logging.DEBUG):
        log.debug("Loaded CGM file %s (%d bytes) for hotspot extraction", path, len(raw))
    return extract_hotspots_from_bytes(raw)


def extract_hotspots_to_directory(
    file_path: str | Path,
    output_dir: str | Path,
    *,
    stem: str = "image",
) -> Path:
    """Extract hotspot regions and write them as JSON into an output directory."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{stem}_0000.hotspots.json"
    hotspots = extract_hotspots(file_path)
    payload = [
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
        for item in hotspots
    ]
    target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if log.isEnabledFor(logging.DEBUG):
        log.debug("Wrote hotspot JSON: %s", target)
    return target


def extract_final_image_and_hotspots(
    file_path: str | Path,
) -> dict[str, object]:
    """Return the final image as SVG text and hotspot dictionaries."""
    path = Path(file_path)
    hotspots = extract_hotspots(path)
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

    return {
        "image": extract_vector_svg(path),
        "hotspots": hotspot_items,
    }


__all__ = [
    "extract_final_image_and_hotspots",
    "extract_hotspots",
    "extract_hotspots_from_bytes",
    "extract_hotspots_to_directory",
]
