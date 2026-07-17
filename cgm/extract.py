"""High-level CGM extraction helpers for SVG composition and hotspot recovery."""

from __future__ import annotations

__author__ = "Kestin Goforth"
__copyright__ = "Copyright 2026"
__license__ = "BSD-3-Clause"

import base64
import html
import io
import json
import logging
import math
import re
import struct
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

try:
    import imagecodecs
except ModuleNotFoundError:
    imagecodecs = None

try:
    from PIL import Image
except ModuleNotFoundError:
    Image = None

from .parser import CELL_ARRAY_CLASS_ID, CELL_ARRAY_ELEMENT_ID, iter_elements
from .types import HotSpot, RawImage

log = logging.getLogger("cgm.extract")

_TEXT_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
_TEXT_HEX_RUN_RE = re.compile(r"[0-9A-Fa-f]{32,}")


def _reverse_bits_in_byte(value: int) -> int:
    """Return a byte with internal bit order reversed."""
    value = ((value & 0xF0) >> 4) | ((value & 0x0F) << 4)
    value = ((value & 0xCC) >> 2) | ((value & 0x33) << 2)
    value = ((value & 0xAA) >> 1) | ((value & 0x55) << 1)
    return value


def _payload_variants(payload: bytes) -> list[bytes]:
    """Return unique encoded payload variants used for decode attempts."""
    variants = [payload]
    reversed_bits = bytes(_reverse_bits_in_byte(byte) for byte in payload)
    if reversed_bits != payload:
        variants.append(reversed_bits)
    return variants


def _candidate_signature(candidate: dict[str, object] | None) -> tuple[str, str, bool] | None:
    """Return normalized signature tuple for a decode candidate entry."""
    if not isinstance(candidate, dict):
        return None
    decoder = candidate.get("decoder")
    variant = candidate.get("encoded_variant")
    invert = candidate.get("invert")
    if not isinstance(decoder, str) or not isinstance(variant, str) or not isinstance(invert, bool):
        return None
    return decoder, variant, invert


def _choose_consensus_decode_signature(
    decoded_tile_meta: dict[int, dict[str, object]],
) -> tuple[str, str, bool] | None:
    """Choose dominant decode signature across all successfully decoded tiles."""
    signatures: list[tuple[str, str, bool]] = []
    for meta in decoded_tile_meta.values():
        candidate = meta.get("best_candidate") if isinstance(meta, dict) else None
        signature = _candidate_signature(candidate if isinstance(candidate, dict) else None)
        if signature is not None:
            signatures.append(signature)

    if not signatures:
        return None

    counts = Counter(signatures)
    return counts.most_common(1)[0][0]


def _choose_consensus_decode_dimensions(
    decoded_tile_meta: dict[int, dict[str, object]],
    *,
    min_fraction: float = 0.0,
    min_count: int = 1,
) -> tuple[int, int] | None:
    """Choose dominant decoder dimensions from successful tile candidates."""
    dimensions: list[tuple[int, int]] = []
    for meta in decoded_tile_meta.values():
        candidate = meta.get("best_candidate") if isinstance(meta, dict) else None
        if not isinstance(candidate, dict):
            continue
        width = candidate.get("width")
        height = candidate.get("height")
        if isinstance(width, int) and isinstance(height, int) and width > 0 and height > 0:
            dimensions.append((width, height))

    if not dimensions:
        return None

    top_dims, top_count = Counter(dimensions).most_common(1)[0]
    if top_count < max(1, min_count):
        return None
    if min_fraction > 0.0 and (top_count / len(dimensions)) < min_fraction:
        return None
    return top_dims


def _parse_cell_array_hints(parameters: bytes) -> tuple[int | None, int | None, int]:
    """Best-effort parse of common Cell Array metadata for binary CGM streams.

    Many CGM files use 16-bit integer VDC coordinates and 16-bit dimensions.
    When this layout is present, this function returns width/height and payload
    offset. Otherwise, it falls back to returning unknown dimensions and a
    conservative payload offset of zero.
    """

    # Typical layout:
    # - P, Q, R points (3 points * 2 coords * 2 bytes) = 12 bytes
    # - nx, ny, local color precision (3 * 2 bytes) = 6 bytes
    # - cell representation mode (2 bytes) = 2 bytes
    # - optional padding/precision-specific details may follow
    base = 20
    if len(parameters) < base:
        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                "Cell Array hint parse skipped: params too short (%d bytes, need >= %d)",
                len(parameters),
                base,
            )
        return None, None, 0

    nx = int.from_bytes(parameters[12:14], "big", signed=False)
    ny = int.from_bytes(parameters[14:16], "big", signed=False)

    if nx == 0 or ny == 0:
        if log.isEnabledFor(logging.DEBUG):
            log.debug("Cell Array hint parse invalid dimensions: nx=%d ny=%d", nx, ny)
        return None, None, 0

    if log.isEnabledFor(logging.DEBUG):
        log.debug("Cell Array hints: width=%d height=%d payload_offset=%d", nx, ny, base)

    return nx, ny, base


def _parse_cell_array_metadata(parameters: bytes) -> dict[str, int | None]:
    """Parse common binary Cell Array metadata in a safe, profile-agnostic way."""
    width, height, payload_offset = _parse_cell_array_hints(parameters)
    if len(parameters) >= 20:
        local_color_precision = int.from_bytes(parameters[16:18], "big", signed=False)
        cell_representation_mode = int.from_bytes(parameters[18:20], "big", signed=False)
    else:
        local_color_precision = None
        cell_representation_mode = None

    return {
        "width": width,
        "height": height,
        "payload_offset": payload_offset,
        "local_color_precision": local_color_precision,
        "cell_representation_mode": cell_representation_mode,
    }


@dataclass(slots=True)
class _SvgStyle:
    stroke_color: str = "#000000"
    stroke_width: float = 1.0
    font_size: float = 10.0
    marker_size: float = 2.0


def _parse_f32_be(value: bytes) -> float | None:
    if len(value) != 4:
        return None
    parsed = float(struct.unpack(">f", value)[0])
    if not math.isfinite(parsed):
        return None
    return parsed


def _decode_point_pairs_exact(parameters: bytes) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for offset in range(0, len(parameters) - 7, 8):
        x = _parse_f32_be(parameters[offset : offset + 4])
        y = _parse_f32_be(parameters[offset + 4 : offset + 8])
        if x is None or y is None:
            continue
        points.append((x, y))
    return points


def _decode_point_pairs_i32(parameters: bytes) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for offset in range(0, len(parameters) - 7, 8):
        x = int.from_bytes(parameters[offset : offset + 4], "big", signed=True)
        y = int.from_bytes(parameters[offset + 4 : offset + 8], "big", signed=True)
        points.append((float(x), float(y)))
    return points


def _decode_point_pairs_heuristic(parameters: bytes) -> list[tuple[float, float]]:
    best: list[tuple[float, float]] = []
    for start in range(8):
        points: list[tuple[float, float]] = []
        for offset in range(start, len(parameters) - 7, 8):
            x = _parse_f32_be(parameters[offset : offset + 4])
            y = _parse_f32_be(parameters[offset + 4 : offset + 8])
            if x is None or y is None:
                continue
            points.append((x, y))
        if len(points) > len(best):
            best = points
    return best


def _decode_cgm_text(parameters: bytes) -> str | None:
    for idx, n_chars in enumerate(parameters):
        if n_chars <= 0:
            continue
        end = idx + 1 + n_chars
        if end != len(parameters):
            continue

        candidate = parameters[idx + 1 : end]
        if any(byte < 32 or byte > 126 for byte in candidate):
            continue
        try:
            return candidate.decode("ascii")
        except UnicodeDecodeError:
            continue
    return None


def _decode_prefixed_ascii(parameters: bytes) -> str | None:
    """Decode a leading length-prefixed ASCII token from bytes."""
    if not parameters:
        return None
    length = parameters[0]
    if length <= 0 or 1 + length > len(parameters):
        return None
    token = parameters[1 : 1 + length]
    if any(byte < 32 or byte > 126 for byte in token):
        return None
    return token.decode("ascii", errors="ignore")


def _decode_application_property(parameters: bytes) -> tuple[str, bytes] | None:
    """Decode class-9/id-1 application data property as key/value bytes."""
    if len(parameters) < 3:
        return None

    key_len = parameters[0]
    if key_len <= 0 or 1 + key_len >= len(parameters):
        return None

    key_bytes = parameters[1 : 1 + key_len]
    if any(byte < 32 or byte > 126 for byte in key_bytes):
        return None
    key = key_bytes.decode("ascii", errors="ignore")
    cursor = 1 + key_len

    # Prefer exact parse with one-byte length, then two-byte length.
    one_len = parameters[cursor]
    one_end = cursor + 1 + one_len
    if one_end == len(parameters):
        return key, parameters[cursor + 1 : one_end]

    if cursor + 2 <= len(parameters):
        two_len = int.from_bytes(parameters[cursor : cursor + 2], "big", signed=False)
        two_end = cursor + 2 + two_len
        if two_end == len(parameters):
            return key, parameters[cursor + 2 : two_end]

    return key, parameters[cursor:]


def _decode_hotspot_region_bbox(value: bytes) -> tuple[int, int, int, int] | None:
    """Extract a best-effort rectangular region from hotspot payload bytes."""
    if len(value) < 8:
        return None
    x_min = int.from_bytes(value[-8:-6], "big", signed=False)
    y_min = int.from_bytes(value[-6:-4], "big", signed=False)
    x_max = int.from_bytes(value[-4:-2], "big", signed=False)
    y_max = int.from_bytes(value[-2:], "big", signed=False)
    if x_min >= x_max or y_min >= y_max:
        return None
    return x_min, y_min, x_max, y_max


def extract_hotspots_from_bytes(data: bytes) -> list[HotSpot]:
    """Extract best-effort hotspot regions from application data elements."""
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
            context_stack.append(_new_context(_decode_prefixed_ascii(element.parameters)))
            continue

        if element.class_id == 9 and element.element_id == 1:
            if not context_stack:
                continue
            prop = _decode_application_property(element.parameters)
            if prop is None:
                continue
            key, value = prop
            context = context_stack[-1]
            if key == "name":
                decoded = value.decode("ascii", errors="ignore").rstrip("\x00")
                context["name"] = decoded or None
            elif key == "region":
                bbox = _decode_hotspot_region_bbox(value)
                if bbox is not None:
                    context["bbox"] = bbox
                    context["region_hex"] = value.hex()
            continue

        if element.class_id == 4 and context_stack:
            context = context_stack[-1]
            if element.element_id in (1, 7):
                for x, y in _decode_point_pairs_exact(element.parameters):
                    _update_geom_bbox(context, x, y)
            elif element.element_id == 5 and len(element.parameters) >= 8:
                x_val = _parse_f32_be(element.parameters[0:4])
                y_val = _parse_f32_be(element.parameters[4:8])
                if x_val is not None and y_val is not None:
                    _update_geom_bbox(context, x_val, y_val)
            elif element.element_id == 29:
                restricted = _decode_restricted_text(element.parameters)
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
    """Extract hotspot regions from a binary CGM file path."""
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
    """Return the final image as SVG text and hotspot dictionaries.

    The returned SVG may contain an embedded PNG image when raster fallback is used.
    """

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


def _default_palette_entry(color_index: int) -> tuple[int, int, int]:
    defaults = {
        0: (0, 0, 0),
        1: (255, 255, 255),
        2: (0, 0, 0),
        3: (47, 47, 47),
        4: (64, 64, 64),
        5: (96, 96, 96),
        6: (128, 128, 128),
        7: (160, 160, 160),
    }
    if color_index in defaults:
        return defaults[color_index]
    gray = max(0, min(255, color_index))
    return gray, gray, gray


def _palette_index_to_hex(
    color_index: int,
    *,
    color_table: dict[int, tuple[int, int, int]] | None = None,
) -> str:
    rgb = (color_table or {}).get(color_index, _default_palette_entry(color_index))
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def _decode_transparency_mode(parameters: bytes) -> bool | None:
    """Return transparency mode (True=on, False=off) when decodable."""
    if not parameters:
        return None
    if len(parameters) >= 2:
        value = int.from_bytes(parameters[:2], "big", signed=False)
    else:
        value = parameters[0]
    return value != 0


def _decode_clip_indicator(parameters: bytes) -> bool | None:
    """Return clip indicator (True=on, False=off) when decodable."""
    if not parameters:
        return None
    if len(parameters) >= 2:
        value = int.from_bytes(parameters[:2], "big", signed=False)
    else:
        value = parameters[0]
    return value != 0


def _decode_color_table_parameters(parameters: bytes) -> dict[int, tuple[int, int, int]]:
    """Decode class-5/id-34 Color Table entries across common precisions."""
    if len(parameters) < 4:
        return {}
    table: dict[int, tuple[int, int, int]] = {}

    # Decode candidate layouts and merge; higher component precision decodes
    # override lower-precision ambiguities for overlapping indices.
    for component_bytes in (1, 2):
        entry_size = component_bytes * 3
        for offset in (1, 2):
            if len(parameters) <= offset:
                continue
            remaining = len(parameters) - offset
            if remaining < entry_size or (remaining % entry_size) != 0:
                continue

            start_index = int.from_bytes(parameters[:offset], "big", signed=False)
            entries = remaining // entry_size
            cursor = offset
            for entry_idx in range(entries):
                if component_bytes == 1:
                    r = parameters[cursor]
                    g = parameters[cursor + 1]
                    b = parameters[cursor + 2]
                else:
                    r16 = int.from_bytes(parameters[cursor : cursor + 2], "big", signed=False)
                    g16 = int.from_bytes(parameters[cursor + 2 : cursor + 4], "big", signed=False)
                    b16 = int.from_bytes(parameters[cursor + 4 : cursor + 6], "big", signed=False)
                    r = _scale_component_to_byte(r16, bits=16)
                    g = _scale_component_to_byte(g16, bits=16)
                    b = _scale_component_to_byte(b16, bits=16)

                table[start_index + entry_idx] = (r, g, b)
                cursor += entry_size

    return table


def _scale_component_to_byte(value: int, *, bits: int) -> int:
    if bits <= 8:
        return max(0, min(255, value))
    max_value = (1 << bits) - 1
    if max_value <= 0:
        return 0
    return max(0, min(255, round((value / max_value) * 255.0)))


def _extract_color_table(data: bytes) -> dict[int, tuple[int, int, int]]:
    table: dict[int, tuple[int, int, int]] = {}
    for element in iter_elements(data):
        if element.class_id == 5 and element.element_id == 34:
            table.update(_decode_color_table_parameters(element.parameters))
    return table


def _decode_color_value_extent_parameters(
    parameters: bytes,
) -> tuple[int, int, int, int, int, int] | None:
    """Decode class-1/id-10 Color Value Extent parameters."""
    if len(parameters) >= 12:
        values = tuple(
            int.from_bytes(parameters[idx : idx + 2], "big", signed=False)
            for idx in (0, 2, 4, 6, 8, 10)
        )
        return values  # type: ignore[return-value]

    if len(parameters) >= 6:
        values = tuple(parameters[idx] for idx in range(6))
        return values  # type: ignore[return-value]

    return None


def _extract_color_value_extent(data: bytes) -> tuple[int, int, int, int, int, int] | None:
    extent: tuple[int, int, int, int, int, int] | None = None
    for element in iter_elements(data):
        if element.class_id == 1 and element.element_id == 10:
            decoded = _decode_color_value_extent_parameters(element.parameters)
            if decoded is not None:
                extent = decoded
    return extent


def _scale_direct16_rgb_payload(
    payload: bytes,
    total_pixels: int,
    *,
    color_value_extent: tuple[int, int, int, int, int, int] | None,
) -> bytes | None:
    required = total_pixels * 6
    if len(payload) < required:
        return None

    if color_value_extent is None:
        mins = (0, 0, 0)
        maxes = (65535, 65535, 65535)
    else:
        mins = color_value_extent[:3]
        maxes = color_value_extent[3:]

    out = bytearray(required // 2)
    cursor = 0
    for px in range(total_pixels):
        base = px * 6
        for ch in range(3):
            value = int.from_bytes(payload[base + (ch * 2) : base + (ch * 2) + 2], "big")
            low = mins[ch]
            high = maxes[ch]
            if high <= low:
                scaled = 255 if value > high else 0
            else:
                bounded = min(max(value, low), high)
                scaled = round(((bounded - low) / (high - low)) * 255.0)
            out[cursor] = max(0, min(255, scaled))
            cursor += 1

    return bytes(out)


def _indexed_palette_bytes(color_table: dict[int, tuple[int, int, int]]) -> bytes:
    channels: list[int] = []
    for index in range(256):
        r, g, b = color_table.get(index, _default_palette_entry(index))
        channels.extend((r, g, b))
    return bytes(channels)


def _format_points(
    points: list[tuple[float, float]],
    *,
    min_x: float,
    max_x: float,
    min_y: float,
    max_y: float,
) -> str:
    mapped = (f"{x:.3f},{_map_svg_y(y, min_y=min_y, max_y=max_y):.3f}" for x, y in points)
    return " ".join(mapped)


def _map_svg_y(y: float, *, min_y: float, max_y: float) -> float:
    return max_y - (y - min_y)


def _filter_points_for_bounds(
    points: list[tuple[float, float]],
    *,
    vdc_extent: tuple[float, float, float, float] | None,
) -> list[tuple[float, float]]:
    """Prefer points near declared VDC extent when computing SVG bounds."""
    if not points or vdc_extent is None:
        return points

    min_x, min_y, max_x, max_y = vdc_extent
    span = max(1.0, max_x - min_x, max_y - min_y)
    margin = span * 0.25
    low_x = min_x - margin
    high_x = max_x + margin
    low_y = min_y - margin
    high_y = max_y + margin

    filtered = [(x, y) for x, y in points if (low_x <= x <= high_x) and (low_y <= y <= high_y)]
    return filtered if filtered else points


def _rebase_points_to_vdc_if_better(
    points: list[tuple[float, float]],
    *,
    vdc_extent: tuple[float, float, float, float] | None,
) -> list[tuple[float, float]]:
    """Shift local-origin point sets into VDC space when clearly better."""
    if not points or vdc_extent is None:
        return points

    min_x, min_y, max_x, max_y = vdc_extent
    original_penalty = _point_penalty(points, vdc_extent=vdc_extent)
    if original_penalty <= 0.0:
        return points

    shifts = [
        (0.0, 0.0),
        (min_x, min_y),
        (min_x, max_y),
        (max_x, min_y),
        (max_x, max_y),
        ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0),
    ]

    best_points = points
    best_penalty = original_penalty
    for dx, dy in shifts:
        candidate = [(x + dx, y + dy) for x, y in points]
        penalty = _point_penalty(candidate, vdc_extent=vdc_extent)
        if penalty < best_penalty:
            best_penalty = penalty
            best_points = candidate

    if best_penalty < (original_penalty * 0.7) and (original_penalty - best_penalty) >= 5.0:
        return best_points
    return points


def _sanitize_points_within_vdc(
    points: list[tuple[float, float]],
    *,
    vdc_extent: tuple[float, float, float, float] | None,
) -> list[tuple[float, float]]:
    """Drop implausible outlier vertices when a stable VDC envelope is known."""
    if len(points) < 2 or vdc_extent is None:
        return points

    min_x, min_y, max_x, max_y = vdc_extent
    span = max(1.0, max_x - min_x, max_y - min_y)
    margin = span * 0.25
    low_x = min_x - margin
    high_x = max_x + margin
    low_y = min_y - margin
    high_y = max_y + margin

    in_range = [(x, y) for x, y in points if (low_x <= x <= high_x) and (low_y <= y <= high_y)]
    if len(in_range) < 2 or len(in_range) == len(points):
        return points

    original_penalty = _point_penalty(points, vdc_extent=vdc_extent)
    sanitized_penalty = _point_penalty(in_range, vdc_extent=vdc_extent)
    if sanitized_penalty + 5.0 <= original_penalty:
        return in_range
    return points


def _extract_vdc_bounded_runs(
    points: list[tuple[float, float]],
    *,
    vdc_extent: tuple[float, float, float, float] | None,
    min_run_points: int = 3,
    max_runs: int = 16,
    max_points_per_run: int = 256,
) -> list[list[tuple[float, float]]]:
    """Split noisy point streams into plausible contiguous VDC-bounded runs."""
    if len(points) < min_run_points:
        return []
    if vdc_extent is None:
        return [points[:max_points_per_run]]

    min_x, min_y, max_x, max_y = vdc_extent
    span = max(1.0, max_x - min_x, max_y - min_y)
    margin = span * 0.25
    low_x = min_x - margin
    high_x = max_x + margin
    low_y = min_y - margin
    high_y = max_y + margin

    runs: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []

    for x, y in points:
        if low_x <= x <= high_x and low_y <= y <= high_y:
            current.append((x, y))
            continue

        if len(current) >= min_run_points:
            runs.append(current[:max_points_per_run])
            if len(runs) >= max_runs:
                return runs
        current = []

    if len(current) >= min_run_points and len(runs) < max_runs:
        runs.append(current[:max_points_per_run])

    return runs


def _decode_i16_be(value: bytes) -> int | None:
    if len(value) != 2:
        return None
    return int.from_bytes(value, "big", signed=True)


def _decode_i32_be(value: bytes) -> int | None:
    if len(value) != 4:
        return None
    return int.from_bytes(value, "big", signed=True)


def _decode_point_pairs_i16(parameters: bytes) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for offset in range(0, len(parameters) - 3, 4):
        x = _decode_i16_be(parameters[offset : offset + 2])
        y = _decode_i16_be(parameters[offset + 2 : offset + 4])
        if x is None or y is None:
            continue
        points.append((float(x), float(y)))
    return points


def _coordinate_candidate_penalty(
    points: list[tuple[float, float]],
    *,
    vdc_extent: tuple[float, float, float, float] | None,
) -> float:
    penalty = _point_penalty(points, vdc_extent=vdc_extent)
    for x, y in points:
        if 0.0 < abs(x) < 1e-20:
            penalty += 10.0
        if 0.0 < abs(y) < 1e-20:
            penalty += 10.0
    return penalty


def _point_penalty(
    points: list[tuple[float, float]],
    *,
    vdc_extent: tuple[float, float, float, float] | None,
) -> float:
    if not points:
        return 1e9

    if vdc_extent is None:
        penalty = 0.0
        for x, y in points:
            if abs(x) > 5_000_000 or abs(y) > 5_000_000:
                penalty += 25.0
            if abs(x) > 50_000_000 or abs(y) > 50_000_000:
                penalty += 250.0
        return penalty

    min_x, min_y, max_x, max_y = vdc_extent
    span = max(1.0, max_x - min_x, max_y - min_y)
    margin = span * 0.25
    low_x = min_x - margin
    high_x = max_x + margin
    low_y = min_y - margin
    high_y = max_y + margin

    penalty = 0.0
    for x, y in points:
        if x < low_x or x > high_x or y < low_y or y > high_y:
            penalty += 5.0
        if x < (min_x - span * 10.0) or x > (max_x + span * 10.0):
            penalty += 50.0
        if y < (min_y - span * 10.0) or y > (max_y + span * 10.0):
            penalty += 50.0

    return penalty


def _decode_point_pairs_best(
    parameters: bytes,
    *,
    vdc_extent: tuple[float, float, float, float] | None,
) -> list[tuple[float, float]]:
    float_points = _decode_point_pairs_exact(parameters)
    i32_points = _decode_point_pairs_i32(parameters)
    i16_points = _decode_point_pairs_i16(parameters)

    candidates = [candidate for candidate in (float_points, i32_points, i16_points) if candidate]
    if not candidates:
        return []

    best = min(
        candidates,
        key=lambda pts: (_coordinate_candidate_penalty(pts, vdc_extent=vdc_extent), -len(pts)),
    )
    return best


def _decode_vdc_extent(parameters: bytes) -> tuple[float, float, float, float] | None:
    if len(parameters) >= 16:
        real_values: list[float] = []
        for idx in range(0, 16, 4):
            value = _parse_f32_be(parameters[idx : idx + 4])
            if value is None:
                real_values = []
                break
            real_values.append(value)
        if len(real_values) == 4:
            x1, y1, x2, y2 = real_values
            if x1 != x2 and y1 != y2:
                return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)

    if len(parameters) >= 8:
        signed_values = [
            int.from_bytes(parameters[idx : idx + 2], "big", signed=True) for idx in (0, 2, 4, 6)
        ]
        x1, y1, x2, y2 = signed_values
        if x1 != x2 and y1 != y2:
            return float(min(x1, x2)), float(min(y1, y2)), float(max(x1, x2)), float(max(y1, y2))

        unsigned_values = [
            int.from_bytes(parameters[idx : idx + 2], "big", signed=False) for idx in (0, 2, 4, 6)
        ]
        x1, y1, x2, y2 = unsigned_values
        if x1 != x2 and y1 != y2:
            return float(min(x1, x2)), float(min(y1, y2)), float(max(x1, x2)), float(max(y1, y2))

    return None


def _decode_xy(
    parameters: bytes,
    *,
    vdc_extent: tuple[float, float, float, float] | None,
) -> tuple[float, float] | None:
    candidates: list[tuple[float, float]] = []
    for offset in (0, 4):
        if len(parameters) < offset + 8:
            continue
        x_f32 = _parse_f32_be(parameters[offset : offset + 4])
        y_f32 = _parse_f32_be(parameters[offset + 4 : offset + 8])
        if x_f32 is not None and y_f32 is not None:
            candidates.append((x_f32, y_f32))

        x_i32 = _decode_i32_be(parameters[offset : offset + 4])
        y_i32 = _decode_i32_be(parameters[offset + 4 : offset + 8])
        if x_i32 is not None and y_i32 is not None:
            candidates.append((float(x_i32), float(y_i32)))

    for offset in (0, 2, 4, 6):
        if len(parameters) < offset + 4:
            continue
        x_i16 = _decode_i16_be(parameters[offset : offset + 2])
        y_i16 = _decode_i16_be(parameters[offset + 2 : offset + 4])
        if x_i16 is not None and y_i16 is not None:
            candidates.append((float(x_i16), float(y_i16)))

    if candidates:
        return min(
            candidates,
            key=lambda pts: _coordinate_candidate_penalty([pts], vdc_extent=vdc_extent),
        )
    return None


def _decode_pairwise_segments(
    parameters: bytes,
    *,
    vdc_extent: tuple[float, float, float, float] | None,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    points = _decode_point_pairs_best(parameters, vdc_extent=vdc_extent)
    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for idx in range(0, len(points) - 1, 2):
        segments.append((points[idx], points[idx + 1]))
    return segments


def _decode_gdp_polyline_points(
    parameters: bytes,
    *,
    vdc_extent: tuple[float, float, float, float] | None,
) -> list[tuple[float, float]]:
    """Best-effort decode for GDP payloads that carry point-like coordinates."""
    best_points: list[tuple[float, float]] = []
    best_rank: tuple[float, int] | None = None

    for offset in (0, 2, 4, 6, 8):
        if offset >= len(parameters):
            break
        payload = parameters[offset:]

        for decoder_bias, decoded in (
            (0.0, _decode_point_pairs_best(payload, vdc_extent=vdc_extent)),
            (0.25, _decode_point_pairs_heuristic(payload)),
        ):
            if len(decoded) < 2:
                continue

            penalty = _point_penalty(decoded, vdc_extent=vdc_extent)
            rank = (penalty + decoder_bias + (offset * 0.1), -len(decoded))
            if best_rank is None or rank < best_rank:
                best_rank = rank
                best_points = decoded

    return best_points


def _decode_rectangle_corners(
    parameters: bytes,
    *,
    vdc_extent: tuple[float, float, float, float] | None,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    points = _decode_point_pairs_best(parameters, vdc_extent=vdc_extent)
    if len(points) >= 2:
        return points[0], points[1]
    return None


def _decode_circle_data(parameters: bytes) -> tuple[float, float, float] | None:
    candidates: list[tuple[float, float, float]] = []
    if len(parameters) >= 12:
        cx_f32 = _parse_f32_be(parameters[0:4])
        cy_f32 = _parse_f32_be(parameters[4:8])
        radius_f32 = _parse_f32_be(parameters[8:12])
        if cx_f32 is not None and cy_f32 is not None and radius_f32 is not None and radius_f32 > 0:
            candidates.append((cx_f32, cy_f32, abs(radius_f32)))

        cx_i32 = _decode_i32_be(parameters[0:4])
        cy_i32 = _decode_i32_be(parameters[4:8])
        radius_i32 = _decode_i32_be(parameters[8:12])
        if cx_i32 is not None and cy_i32 is not None and radius_i32 is not None and radius_i32 > 0:
            candidates.append((float(cx_i32), float(cy_i32), float(radius_i32)))

    if len(parameters) >= 6:
        cx_i16 = _decode_i16_be(parameters[0:2])
        cy_i16 = _decode_i16_be(parameters[2:4])
        radius_i16 = _decode_i16_be(parameters[4:6])
        if cx_i16 is not None and cy_i16 is not None and radius_i16 is not None and radius_i16 > 0:
            candidates.append((float(cx_i16), float(cy_i16), float(radius_i16)))

    if candidates:
        return min(
            candidates,
            key=lambda candidate: _coordinate_candidate_penalty(
                [(candidate[0], candidate[1]), (candidate[0], candidate[1] + candidate[2])],
                vdc_extent=None,
            ),
        )

    return None


def _decode_ellipse_data(
    parameters: bytes,
    *,
    vdc_extent: tuple[float, float, float, float] | None,
) -> tuple[float, float, float, float] | None:
    points = _decode_point_pairs_best(parameters, vdc_extent=vdc_extent)
    if len(points) < 3:
        return None

    center = points[0]
    major = points[1]
    minor = points[2]
    rx = ((major[0] - center[0]) ** 2 + (major[1] - center[1]) ** 2) ** 0.5
    ry = ((minor[0] - center[0]) ** 2 + (minor[1] - center[1]) ** 2) ** 0.5
    if rx <= 0 or ry <= 0:
        return None

    return center[0], center[1], rx, ry


def _decode_restricted_text(parameters: bytes) -> tuple[float, float, float, float, str] | None:
    """Decode CGM class-4/id-29 Restricted Text (profile-specific).

    Expected parameter order:
    - box_width, box_height
    - anchor_x, anchor_y
    - text string payload
    """

    if len(parameters) < 10:
        return None

    for coord_size in (2, 4):
        fixed_size = coord_size * 4
        if len(parameters) <= fixed_size:
            continue

        coord_bytes = parameters[:fixed_size]
        text_bytes = parameters[fixed_size:]
        text = _decode_cgm_text(text_bytes)
        if not text:
            continue

        signed = coord_size == 4
        box_w = int.from_bytes(coord_bytes[0:coord_size], "big", signed=signed)
        box_h = int.from_bytes(coord_bytes[coord_size : 2 * coord_size], "big", signed=signed)
        anchor_x = int.from_bytes(
            coord_bytes[2 * coord_size : 3 * coord_size], "big", signed=signed
        )
        anchor_y = int.from_bytes(
            coord_bytes[3 * coord_size : 4 * coord_size], "big", signed=signed
        )

        if box_w == 0 or box_h == 0:
            continue

        return float(anchor_x), float(anchor_y), float(box_w), float(box_h), text

    return None


def _ascii_printable_ratio(data: bytes) -> float:
    """Return ratio of printable ASCII bytes in payload."""
    if not data:
        return 0.0
    printable = sum(32 <= byte <= 126 for byte in data)
    return printable / len(data)


def _shannon_entropy(data: bytes) -> float:
    """Compute Shannon entropy in bits per byte."""
    if not data:
        return 0.0
    counts = Counter(data)
    total = len(data)
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


def _find_ascii_runs(data: bytes, *, min_len: int = 4, limit: int = 8) -> list[dict[str, object]]:
    """Find short printable ASCII runs within payload for diagnostics."""
    runs: list[dict[str, object]] = []
    idx = 0
    while idx < len(data) and len(runs) < limit:
        if 32 <= data[idx] <= 126:
            end = idx + 1
            while end < len(data) and 32 <= data[end] <= 126:
                end += 1
            if end - idx >= min_len:
                preview = data[idx:end][:48].decode("ascii", errors="ignore")
                runs.append(
                    {
                        "offset": idx,
                        "length": end - idx,
                        "preview": preview,
                    }
                )
            idx = end
        else:
            idx += 1
    return runs


def _leading_run_length(data: bytes, *, value: int) -> int:
    """Return run-length of a given byte at payload start."""
    count = 0
    for byte in data:
        if byte != value:
            break
        count += 1
    return count


def _analyze_element29_payload(parameters: bytes) -> dict[str, object]:
    """Provide structured diagnostics for class-4/id-29 payloads."""
    byte_counts = Counter(parameters)
    top_bytes = [
        {"byte": f"0x{value:02x}", "count": count} for value, count in byte_counts.most_common(8)
    ]
    head_u16 = [
        int.from_bytes(parameters[idx : idx + 2], "big")
        for idx in range(0, min(64, len(parameters) - 1), 2)
    ]
    restricted = _decode_restricted_text(parameters)
    ascii_ratio = _ascii_printable_ratio(parameters)
    entropy = _shannon_entropy(parameters)

    return {
        "length": len(parameters),
        "ascii_ratio": round(ascii_ratio, 4),
        "entropy_bits_per_byte": round(entropy, 4),
        "top_bytes": top_bytes,
        "leading_ff_run": _leading_run_length(parameters, value=0xFF),
        "ascii_runs": _find_ascii_runs(parameters),
        "head_u16": head_u16,
        "restricted_text_detected": restricted is not None,
        "likely_binary_payload": ascii_ratio < 0.45 and entropy > 6.5,
    }


def _collect_element29_decode_candidates(
    parameters: bytes,
    *,
    limit: int = 8,
) -> list[dict[str, object]]:
    """Collect top decode candidates for class-4/id-29 payload diagnostics."""

    if imagecodecs is None or not parameters:
        return []

    candidate_offsets = [
        *range(0, 33),
        157,
        158,
        159,
        160,
        161,
        176,
        184,
        188,
        192,
        196,
        200,
        208,
        224,
        240,
        256,
    ]
    candidate_sizes = [
        (768, 1099),
        (767, 1099),
        (768, 1100),
        (576, 768),
        (575, 767),
    ]

    ranked: list[tuple[float, bool, float, dict[str, object]]] = []

    for offset in candidate_offsets:
        payload = parameters[offset:]
        if not payload:
            continue

        for width, height in candidate_sizes:
            decode_attempts: list[tuple[bytes, str]] = []
            try:
                decode_attempts.append(
                    (
                        bytes(imagecodecs.ccittfax4_decode(payload, height=height, width=width)),
                        "fax4",
                    )
                )
            except (RuntimeError, ValueError):
                pass

            for t4options in (0, 2, 4, 6, 1, 3, 5, 7):
                try:
                    decoded = imagecodecs.ccittfax3_decode(
                        payload,
                        height=height,
                        width=width,
                        t4options=t4options,
                    )
                    decode_attempts.append((bytes(decoded), f"fax3:{t4options}"))
                except (RuntimeError, ValueError):
                    continue

            for decoded, decoder_name in decode_attempts:
                bits = _decode_fax_output_to_bitmap(decoded, width, height)
                if bits is None:
                    continue

                for invert in (False, True):
                    candidate = [1 - bit for bit in bits] if invert else bits
                    score = _score_bitmap(candidate, width, height)
                    black_ratio = sum(candidate) / len(candidate)
                    low_confidence = _is_low_confidence_element29_bitmap(candidate, width, height)

                    candidate_info = {
                        "offset": offset,
                        "width": width,
                        "height": height,
                        "decoder": decoder_name,
                        "invert": invert,
                        "score": round(score, 6),
                        "black_ratio": round(black_ratio, 6),
                        "low_confidence": low_confidence,
                    }
                    ranked.append((score, low_confidence, abs(black_ratio - 0.5), candidate_info))

    ranked.sort(key=lambda item: (item[0], item[1], item[2]))
    return [entry for *_meta, entry in ranked[: max(1, limit)]]


def _decode_fax_output_to_bitmap(output: bytes, width: int, height: int) -> list[int] | None:
    """Normalize CCITT decoder output to a 0/1 bitmap list.

    Some builds return one byte per pixel, while others return packed 1-bit rows.
    """

    total = width * height
    if total <= 0:
        return None

    if len(output) >= total:
        sample = output[:total]
        # Byte-per-pixel: map non-zero values to black=1, zero to white=0.
        return [1 if value else 0 for value in sample]

    row_bytes = (width + 7) // 8
    packed_needed = row_bytes * height
    if len(output) < packed_needed:
        return None

    bits: list[int] = []
    packed = output[:packed_needed]
    for y in range(height):
        row_start = y * row_bytes
        for x in range(width):
            byte_value = packed[row_start + (x // 8)]
            bit = (byte_value >> (7 - (x % 8))) & 1
            bits.append(bit)
    return bits


def _score_bitmap(bits: list[int], width: int, height: int) -> float:
    """Return a polarity-agnostic plausibility score for decoded bitmaps."""

    total = len(bits)
    if total == 0:
        return 1e9

    black = sum(bits)
    black_ratio = black / total
    dominant_ratio = max(black_ratio, 1.0 - black_ratio)
    # Reject nearly single-color outputs regardless of polarity.
    if dominant_ratio >= 0.999:
        return 1e8

    h_steps = max(1, height // 200)
    v_steps = max(1, width // 200)
    h_transitions = 0
    v_transitions = 0

    for y in range(0, height, h_steps):
        row = y * width
        prev = bits[row]
        for x in range(1, width):
            cur = bits[row + x]
            if cur != prev:
                h_transitions += 1
            prev = cur

    for x in range(0, width, v_steps):
        prev = bits[x]
        for y in range(1, height):
            cur = bits[y * width + x]
            if cur != prev:
                v_transitions += 1
            prev = cur

    h_samples = max(1, ((height + h_steps - 1) // h_steps) * max(1, width - 1))
    v_samples = max(1, ((width + v_steps - 1) // v_steps) * max(1, height - 1))
    h_density = h_transitions / h_samples
    v_density = v_transitions / v_samples

    if max(h_density, v_density) < 0.001:
        return 1e8

    transition_ratio = min(h_density, v_density) / max(1e-9, h_density, v_density)

    # Prefer bitmaps with meaningful and reasonably balanced structure in both axes.
    score = 1.0 - transition_ratio
    if transition_ratio < 0.2:
        score += 0.8
    elif transition_ratio < 0.35:
        score += 0.3

    if max(h_density, v_density) < 0.01:
        score += 0.3
    return score


def _is_low_confidence_element29_bitmap(bits: list[int], width: int, height: int) -> bool:
    """Detect common low-confidence decode artifacts (top-band garbage).

    This check is polarity-agnostic by evaluating the minority pixel class. It
    rejects outputs where sparse detail is concentrated in only the top portion
    of the frame, which commonly indicates a wrong offset/dimension decode.
    """

    total = width * height
    if total <= 0 or len(bits) != total:
        return True

    ones = sum(bits)
    zeros = total - ones
    minority_value = 1 if ones <= zeros else 0
    minority_count = min(ones, zeros)
    minority_ratio = minority_count / total

    # If detail density is not sparse, keep it.
    if minority_ratio > 0.08:
        return False

    min_y = height
    max_y = -1
    for idx, value in enumerate(bits):
        if value != minority_value:
            continue
        y = idx // width
        if y < min_y:
            min_y = y
        if y > max_y:
            max_y = y

    if max_y < 0:
        return True

    band_height = max_y - min_y + 1
    band_coverage = band_height / height
    if max_y < int(height * 0.6):
        return True
    if band_coverage < 0.5:
        return True

    return False


def _bitmap_to_png_data_uri(bits: list[int], width: int, height: int) -> str | None:
    """Encode a 0/1 bitmap to a PNG data URI if Pillow is available."""

    if Image is None:
        return None

    if len(bits) != width * height:
        return None

    # Render black lines on white background (user screenshot indicates possible inversion).
    pixels = bytes(0 if bit else 255 for bit in bits)
    image = Image.frombytes("L", (width, height), pixels)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def _image_to_png_data_uri(image: Image.Image | None) -> str | None:
    if Image is None or image is None:
        return None

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def _decode_element29_binary_raster(
    parameters: bytes,
    *,
    allow_low_confidence: bool = False,
) -> tuple[str, int, int] | None:
    """Best-effort decode for binary class-4/id-29 payloads using CCITT codecs."""

    if imagecodecs is None:
        return None

    candidate_offsets = [
        *range(0, 33),
        157,
        158,
        159,
        160,
        161,
        176,
        184,
        188,
        192,
        196,
        200,
        208,
        224,
        240,
        256,
    ]
    candidate_sizes = [
        (768, 1099),
        (767, 1099),
        (768, 1100),
        (576, 768),
        (575, 767),
    ]

    best_score = 1e9
    best_bits: list[int] | None = None
    best_size: tuple[int, int] | None = None
    best_black_ratio = -1.0

    for offset in candidate_offsets:
        payload = parameters[offset:]
        if not payload:
            continue

        for width, height in candidate_sizes:
            # Try Group 4 first, then Group 3 options.
            decode_attempts: list[tuple[bytes, str]] = []
            try:
                decode_attempts.append(
                    (
                        bytes(imagecodecs.ccittfax4_decode(payload, height=height, width=width)),
                        "fax4",
                    )
                )
            except (RuntimeError, ValueError):
                pass

            for t4options in (0, 2, 4, 6, 1, 3, 5, 7):
                try:
                    decoded = imagecodecs.ccittfax3_decode(
                        payload,
                        height=height,
                        width=width,
                        t4options=t4options,
                    )
                    decode_attempts.append((bytes(decoded), f"fax3:{t4options}"))
                except (RuntimeError, ValueError):
                    continue

            for decoded, _decoder_name in decode_attempts:
                bits = _decode_fax_output_to_bitmap(decoded, width, height)
                if bits is None:
                    continue

                for invert in (False, True):
                    candidate = [1 - bit for bit in bits] if invert else bits
                    score = _score_bitmap(candidate, width, height)
                    black_ratio = sum(candidate) / len(candidate)
                    if score < best_score:
                        best_score = score
                        best_bits = candidate
                        best_size = (width, height)
                        best_black_ratio = black_ratio
                    elif abs(score - best_score) <= 1e-9 and black_ratio > best_black_ratio:
                        # Polarity-agnostic scoring can tie exact inversions; prefer dark canvas
                        # fallback for these binary element-29 payloads so white detail remains
                        # visible.
                        best_bits = candidate
                        best_size = (width, height)
                        best_black_ratio = black_ratio

    if best_bits is None or best_size is None:
        return None

    width, height = best_size
    if _is_low_confidence_element29_bitmap(best_bits, width, height):
        if allow_low_confidence:
            if log.isEnabledFor(logging.DEBUG):
                log.debug(
                    "Accepted low-confidence class-4/id-29 raster decode (%dx%d)",
                    width,
                    height,
                )
        else:
            if log.isEnabledFor(logging.DEBUG):
                log.debug(
                    "Rejected low-confidence class-4/id-29 raster decode (%dx%d)",
                    width,
                    height,
                )
            return None

    data_uri = _bitmap_to_png_data_uri(best_bits, width, height)
    if data_uri is None:
        return None
    return data_uri, width, height


def _decode_best_element29_fallback(
    payloads: list[bytes],
) -> tuple[str, int, int] | None:
    """Decode class-4/id-29 payloads, preferring confident candidates first."""
    for payload in payloads:
        decoded = _decode_element29_binary_raster(payload, allow_low_confidence=False)
        if decoded is not None:
            return decoded

    for payload in payloads:
        decoded = _decode_element29_binary_raster(payload, allow_low_confidence=True)
        if decoded is not None:
            return decoded

    return None


def extract_vector_svg_from_bytes(data: bytes) -> str:
    """Convert vector-like CGM primitives from bytes into a best-effort SVG string."""
    style = _SvgStyle()
    color_table = _extract_color_table(data)
    text_items: list[tuple[float, float, str, str, float]] = []
    restricted_text_items: list[tuple[float, float, float, float, str, str, float]] = []
    polyline_items: list[tuple[list[tuple[float, float]], str, float]] = []
    polygon_items: list[tuple[list[tuple[float, float]], str, float]] = []
    marker_items: list[tuple[float, float, str, float]] = []
    rectangle_items: list[tuple[float, float, float, float, str, float]] = []
    circle_items: list[tuple[float, float, float, str, float]] = []
    ellipse_items: list[tuple[float, float, float, float, str, float]] = []
    gdp_parameters: list[bytes] = []
    element29_payloads: list[bytes] = []
    vdc_extent: tuple[float, float, float, float] | None = None
    raster_tile_overlays = _render_first_tile_array_overlays(data)
    raster_background = (
        None if raster_tile_overlays is not None else _render_raster_background_data_uri(data)
    )
    transparency_enabled = True
    clip_enabled = False
    clip_rectangle: tuple[float, float, float, float] | None = None

    for element in iter_elements(data):
        if element.class_id == 2 and element.element_id == 6:
            decoded_vdc = _decode_vdc_extent(element.parameters)
            if decoded_vdc is not None:
                vdc_extent = decoded_vdc
            continue

        if element.class_id == 3 and element.element_id == 4:
            mode = _decode_transparency_mode(element.parameters)
            if isinstance(mode, bool):
                transparency_enabled = mode
            continue

        if element.class_id == 3 and element.element_id == 5:
            decoded_clip = _decode_vdc_extent(element.parameters)
            if decoded_clip is not None:
                clip_rectangle = decoded_clip
            continue

        if element.class_id == 3 and element.element_id == 6:
            indicator = _decode_clip_indicator(element.parameters)
            if isinstance(indicator, bool):
                clip_enabled = indicator
            continue

        if element.class_id == 5:
            if element.element_id == 3 and len(element.parameters) >= 4:
                width = _parse_f32_be(element.parameters[:4])
                if width is not None and width > 0:
                    style.stroke_width = max(0.25, width)
            elif element.element_id == 4 and element.parameters:
                style.stroke_color = _palette_index_to_hex(
                    element.parameters[0],
                    color_table=color_table,
                )
            elif element.element_id == 15 and len(element.parameters) >= 4:
                char_height = _parse_f32_be(element.parameters[:4])
                if char_height is not None and char_height > 0:
                    style.font_size = max(6.0, char_height)
            continue

        if element.class_id != 4:
            continue

        if element.element_id == 1:
            points = _decode_point_pairs_best(element.parameters, vdc_extent=vdc_extent)
            if len(points) >= 2:
                polyline_items.append((points, style.stroke_color, style.stroke_width))
        elif element.element_id == 2:
            for start, end in _decode_pairwise_segments(element.parameters, vdc_extent=vdc_extent):
                polyline_items.append(([start, end], style.stroke_color, style.stroke_width))
        elif element.element_id == 3:
            points = _decode_point_pairs_best(element.parameters, vdc_extent=vdc_extent)
            for marker_x, marker_y in points:
                marker_items.append((marker_x, marker_y, style.stroke_color, style.marker_size))
        elif element.element_id == 4:
            appended = _decode_cgm_text(element.parameters)
            if appended and text_items:
                last_x, last_y, last_text, last_color, last_font = text_items[-1]
                text_items[-1] = (last_x, last_y, f"{last_text}{appended}", last_color, last_font)
        elif element.element_id == 7:
            points = _decode_point_pairs_best(element.parameters, vdc_extent=vdc_extent)
            if len(points) >= 3:
                polygon_items.append((points, style.stroke_color, style.stroke_width))
        elif element.element_id == 8:
            points = _decode_point_pairs_best(element.parameters, vdc_extent=vdc_extent)
            if len(points) >= 3:
                polygon_items.append((points, style.stroke_color, style.stroke_width))
        elif element.element_id == 5:
            xy = _decode_xy(element.parameters, vdc_extent=vdc_extent)
            if xy is None:
                continue
            x, y = xy
            text = _decode_cgm_text(element.parameters[8:])
            if text is None:
                text = _decode_cgm_text(element.parameters[4:])
            if text:
                text_items.append((x, y, text, style.stroke_color, style.font_size))
        elif element.element_id == 6:
            appended = _decode_cgm_text(element.parameters)
            if appended and text_items:
                last_x, last_y, last_text, last_color, last_font = text_items[-1]
                text_items[-1] = (last_x, last_y, f"{last_text}{appended}", last_color, last_font)
        elif element.element_id in (10, 26):
            gdp_parameters.append(element.parameters)
        elif element.element_id == 11:
            corners = _decode_rectangle_corners(element.parameters, vdc_extent=vdc_extent)
            if corners is None:
                continue
            (x1, y1), (x2, y2) = corners
            rectangle_items.append(
                (
                    min(x1, x2),
                    min(y1, y2),
                    abs(x2 - x1),
                    abs(y2 - y1),
                    style.stroke_color,
                    style.stroke_width,
                )
            )
        elif element.element_id == 12:
            circle = _decode_circle_data(element.parameters)
            if circle is None:
                continue
            cx, cy, radius = circle
            circle_items.append((cx, cy, radius, style.stroke_color, style.stroke_width))
        elif element.element_id in (13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 27):
            arc_points = _decode_point_pairs_best(element.parameters, vdc_extent=vdc_extent)
            if len(arc_points) >= 2:
                polyline_items.append((arc_points, style.stroke_color, style.stroke_width))
        elif element.element_id == 28:
            # Element 28 payloads are profile-specific and often include packed
            # control bytes mixed with coordinates. Recover only plausible
            # contiguous runs inside the declared VDC envelope.
            raw_points = _decode_point_pairs_i16(element.parameters)
            for run in _extract_vdc_bounded_runs(raw_points, vdc_extent=vdc_extent):
                polyline_items.append((run, style.stroke_color, style.stroke_width))
        elif element.element_id == 9:
            # Cell Array payloads are handled through raster background extraction.
            continue
        elif element.element_id == 17:
            ellipse = _decode_ellipse_data(element.parameters, vdc_extent=vdc_extent)
            if ellipse is None:
                continue
            cx, cy, rx, ry = ellipse
            ellipse_items.append((cx, cy, rx, ry, style.stroke_color, style.stroke_width))
        elif element.element_id == 29:
            restricted = _decode_restricted_text(element.parameters)
            if restricted is not None:
                anchor_x, anchor_y, box_w, box_h, text = restricted
                restricted_text_items.append(
                    (anchor_x, anchor_y, box_w, box_h, text, style.stroke_color, style.font_size)
                )
            else:
                element29_payloads.append(element.parameters)

    for parameters in gdp_parameters:
        points = _decode_gdp_polyline_points(parameters, vdc_extent=vdc_extent)
        if len(points) >= 2:
            polyline_items.append((points, style.stroke_color, style.stroke_width))

    all_points: list[tuple[float, float]] = []
    for points, _, _ in polyline_items:
        all_points.extend(points)
    for points, _, _ in polygon_items:
        all_points.extend(points)
    for x, y, _, _ in marker_items:
        all_points.append((x, y))
    for x, y, rect_w, rect_h, _, _ in rectangle_items:
        all_points.append((x, y))
        all_points.append((x + rect_w, y + rect_h))
    for cx, cy, radius, _, _ in circle_items:
        all_points.append((cx - radius, cy - radius))
        all_points.append((cx + radius, cy + radius))
    for cx, cy, rx, ry, _, _ in ellipse_items:
        all_points.append((cx - rx, cy - ry))
        all_points.append((cx + rx, cy + ry))
    for x, y, _, _, _ in text_items:
        all_points.append((x, y))
    for x, y, box_w, box_h, _, _, _ in restricted_text_items:
        all_points.append((x, y))
        all_points.append((x + box_w, y + box_h))

    if all_points:
        bounds_points = _filter_points_for_bounds(all_points, vdc_extent=vdc_extent)
        min_x = min(point[0] for point in bounds_points)
        max_x = max(point[0] for point in bounds_points)
        min_y = min(point[1] for point in bounds_points)
        max_y = max(point[1] for point in bounds_points)
    elif vdc_extent is not None:
        min_x, min_y, max_x, max_y = vdc_extent
    elif raster_tile_overlays is not None:
        _tile_entries, raster_w, raster_h = raster_tile_overlays
        min_x, min_y, max_x, max_y = 0.0, 0.0, float(raster_w), float(raster_h)
    elif raster_background is not None:
        _href, raster_w, raster_h = raster_background
        min_x, min_y, max_x, max_y = 0.0, 0.0, float(raster_w), float(raster_h)
    else:
        min_x, min_y, max_x, max_y = 0.0, 0.0, 100.0, 100.0

    min_x = round(min_x, 3)
    min_y = round(min_y, 3)
    max_x = round(max_x, 3)
    max_y = round(max_y, 3)

    width = max(1.0, round(max_x - min_x, 3))
    height = max(1.0, round(max_y - min_y, 3))

    svg_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'viewBox="{min_x:.3f} {min_y:.3f} {width:.3f} {height:.3f}">'
        ),
    ]

    if not transparency_enabled:
        svg_lines.append(
            f'  <rect x="{min_x:.3f}" y="{min_y:.3f}" width="{width:.3f}" '
            f'height="{height:.3f}" fill="#ffffff" stroke="none" />'
        )

    clip_id = "cgmClip0"
    clip_attr = ""
    if clip_enabled and clip_rectangle is not None:
        clip_x1, clip_y1, clip_x2, clip_y2 = clip_rectangle
        clip_min_x = min(clip_x1, clip_x2)
        clip_max_x = max(clip_x1, clip_x2)
        clip_min_y = min(clip_y1, clip_y2)
        clip_max_y = max(clip_y1, clip_y2)
        clip_w = max(0.0, clip_max_x - clip_min_x)
        clip_h = max(0.0, clip_max_y - clip_min_y)
        if clip_w > 0.0 and clip_h > 0.0:
            clip_svg_y = _map_svg_y(clip_max_y, min_y=min_y, max_y=max_y)
            svg_lines.append("  <defs>")
            svg_lines.append(f'    <clipPath id="{clip_id}">')
            svg_lines.append(
                f'      <rect x="{clip_min_x:.3f}" y="{clip_svg_y:.3f}" '
                f'width="{clip_w:.3f}" height="{clip_h:.3f}" />'
            )
            svg_lines.append("    </clipPath>")
            svg_lines.append("  </defs>")
            clip_attr = f' clip-path="url(#{clip_id})"'

    if raster_tile_overlays is not None:
        tile_entries, tile_total_w, tile_total_h = raster_tile_overlays
        scale_x = width / max(1.0, float(tile_total_w))
        scale_y = height / max(1.0, float(tile_total_h))
        for href, tile_x, tile_y, tile_w, tile_h in tile_entries:
            mapped_x = min_x + (tile_x * scale_x)
            mapped_y = min_y + (tile_y * scale_y)
            mapped_w = max(0.0, tile_w * scale_x)
            mapped_h = max(0.0, tile_h * scale_y)
            if mapped_w <= 0.0 or mapped_h <= 0.0:
                continue
            svg_lines.append(
                f'  <image x="{mapped_x:.3f}" y="{mapped_y:.3f}" '
                f'width="{mapped_w:.3f}" height="{mapped_h:.3f}" preserveAspectRatio="none" '
                f'image-rendering="pixelated" href="{href}"{clip_attr} />'
            )
    elif raster_background is not None:
        href, _raster_w, _raster_h = raster_background
        svg_lines.append(
            f'  <image x="{min_x:.3f}" y="{min_y:.3f}" width="{width:.3f}" '
            f'height="{height:.3f}" preserveAspectRatio="xMidYMid meet" '
            f'image-rendering="pixelated" href="{href}"{clip_attr} />'
        )

    svg_lines.append(f'  <g fill="none" stroke-linecap="round" stroke-linejoin="round"{clip_attr}>')

    for points, stroke_color, stroke_width in polyline_items:
        point_str = _format_points(
            points,
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
        )
        svg_lines.append(
            f'    <polyline points="{point_str}" stroke="{stroke_color}" '
            f'stroke-width="{stroke_width:.3f}" />'
        )

    for points, stroke_color, stroke_width in polygon_items:
        point_str = _format_points(
            points,
            min_x=min_x,
            max_x=max_x,
            min_y=min_y,
            max_y=max_y,
        )
        svg_lines.append(
            f'    <polygon points="{point_str}" stroke="{stroke_color}" '
            f'stroke-width="{stroke_width:.3f}" fill="none" />'
        )

    for x, y, marker_color, marker_size in marker_items:
        svg_lines.append(
            f'    <circle cx="{x:.3f}" cy="{_map_svg_y(y, min_y=min_y, max_y=max_y):.3f}" '
            f'r="{marker_size:.3f}" fill="{marker_color}" stroke="none" />'
        )

    for x, y, rect_w, rect_h, stroke_color, stroke_width in rectangle_items:
        svg_lines.append(
            f'    <rect x="{x:.3f}" y="{_map_svg_y(y + rect_h, min_y=min_y, max_y=max_y):.3f}" '
            f'width="{rect_w:.3f}" height="{rect_h:.3f}" stroke="{stroke_color}" '
            f'stroke-width="{stroke_width:.3f}" fill="none" />'
        )

    for cx, cy, radius, stroke_color, stroke_width in circle_items:
        svg_lines.append(
            f'    <circle cx="{cx:.3f}" cy="{_map_svg_y(cy, min_y=min_y, max_y=max_y):.3f}" '
            f'r="{radius:.3f}" stroke="{stroke_color}" stroke-width="{stroke_width:.3f}" '
            'fill="none" />'
        )

    for cx, cy, rx, ry, stroke_color, stroke_width in ellipse_items:
        svg_lines.append(
            f'    <ellipse cx="{cx:.3f}" cy="{_map_svg_y(cy, min_y=min_y, max_y=max_y):.3f}" '
            f'rx="{rx:.3f}" ry="{ry:.3f}" stroke="{stroke_color}" '
            f'stroke-width="{stroke_width:.3f}" fill="none" />'
        )

    svg_lines.append("  </g>")

    for x, y, text, stroke_color, font_size in text_items:
        mapped_y = _map_svg_y(y, min_y=min_y, max_y=max_y)
        escaped = html.escape(text)
        svg_lines.append(
            f'  <text x="{x:.3f}" y="{mapped_y:.3f}" '
            f'fill="{stroke_color}" font-size="{font_size:.3f}" '
            f'font-family="sans-serif"{clip_attr}>{escaped}</text>'
        )

    for x, y, box_w, box_h, text, stroke_color, font_size in restricted_text_items:
        mapped_y = _map_svg_y(y, min_y=min_y, max_y=max_y)
        box_width = max(1.0, abs(box_w))
        capped_font = max(4.0, min(font_size, abs(box_h)))
        escaped = html.escape(text)
        svg_lines.append(
            f'  <text x="{x:.3f}" y="{mapped_y:.3f}" '
            f'fill="{stroke_color}" font-size="{capped_font:.3f}" '
            f'font-family="sans-serif" textLength="{box_width:.3f}" '
            f'lengthAdjust="spacingAndGlyphs"{clip_attr}>{escaped}</text>'
        )

    if (
        not polyline_items
        and not polygon_items
        and not marker_items
        and not rectangle_items
        and not circle_items
        and not ellipse_items
        and not text_items
        and not restricted_text_items
        and raster_background is None
        and element29_payloads
    ):
        decoded = _decode_best_element29_fallback(element29_payloads)
        if decoded is not None:
            href, raster_width, raster_height = decoded
            svg_lines.append(
                f'  <image x="{min_x:.3f}" y="{min_y:.3f}" width="{width:.3f}" '
                f'height="{height:.3f}" preserveAspectRatio="xMidYMid meet" '
                f'image-rendering="pixelated" href="{href}"{clip_attr} />'
            )
            if log.isEnabledFor(logging.DEBUG):
                log.debug(
                    "Rendered class-4/id-29 payload as raster image (%dx%d)",
                    raster_width,
                    raster_height,
                )

    svg_lines.append("</svg>")
    return "\n".join(svg_lines) + "\n"


def extract_vector_svg(file_path: str | Path) -> str:
    """Convert a CGM file to a best-effort SVG document string."""
    path = Path(file_path)
    raw = path.read_bytes()
    if log.isEnabledFor(logging.DEBUG):
        log.debug("Loaded CGM file %s (%d bytes) for vector SVG conversion", path, len(raw))
    return extract_vector_svg_from_bytes(raw)


def extract_vector_svg_to_directory(
    file_path: str | Path,
    output_dir: str | Path,
    *,
    stem: str = "image",
) -> Path:
    """Convert a CGM file to SVG and write it to an output directory."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{stem}_0000.svg"
    svg = extract_vector_svg(file_path)
    target.write_text(svg, encoding="utf-8")
    if log.isEnabledFor(logging.DEBUG):
        log.debug("Wrote vector SVG: %s", target)
    return target


def _build_data_snapshot(data: bytes) -> dict[str, object]:
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
            candidates = _collect_element29_decode_candidates(element.parameters)
            element29_analysis.append(
                {
                    "offset": element.offset,
                    **_analyze_element29_payload(element.parameters),
                    "decode_candidates": candidates,
                    "has_plausible_decode": any(
                        not item.get("low_confidence", True) for item in candidates
                    ),
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
        "vector_svg": extract_vector_svg_from_bytes(data),
    }


def extract_data_json_from_bytes(data: bytes) -> str:
    """Serialize parsed CGM content, extracted payloads, and SVG into JSON."""
    snapshot = _build_data_snapshot(data)
    return json.dumps(snapshot, indent=2)


def extract_data_json(file_path: str | Path) -> str:
    """Load a CGM file and serialize parsed data to JSON."""
    path = Path(file_path)
    raw = path.read_bytes()
    if log.isEnabledFor(logging.DEBUG):
        log.debug("Loaded CGM file %s (%d bytes) for JSON export", path, len(raw))
    return extract_data_json_from_bytes(raw)


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
    json_text = extract_data_json(file_path)
    target.write_text(json_text + "\n", encoding="utf-8")
    if log.isEnabledFor(logging.DEBUG):
        log.debug("Wrote data JSON: %s", target)
    return target


def extract_raw_images_from_bytes(data: bytes) -> list[RawImage]:
    """Extract raw raster payloads from Cell Array elements in a CGM stream."""
    images: list[RawImage] = []
    image_index = 0
    element_histogram: Counter[tuple[int, int]] = Counter()

    if log.isEnabledFor(logging.DEBUG):
        log.debug("Starting extraction from %d bytes", len(data))

    for element in iter_elements(data):
        element_key = (element.class_id, element.element_id)
        element_histogram[element_key] += 1

        if element.class_id != CELL_ARRAY_CLASS_ID or element.element_id != CELL_ARRAY_ELEMENT_ID:
            continue

        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                "Found Cell Array element: offset=%d param_length=%d",
                element.offset,
                len(element.parameters),
            )

        metadata = _parse_cell_array_metadata(element.parameters)
        width = metadata["width"] if isinstance(metadata["width"], int) else None
        height = metadata["height"] if isinstance(metadata["height"], int) else None
        payload_offset = (
            metadata["payload_offset"] if isinstance(metadata["payload_offset"], int) else 0
        )
        payload = (
            element.parameters[payload_offset:] if payload_offset < len(element.parameters) else b""
        )

        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                (
                    "Extracted image payload: index=%d offset=%d width=%s height=%s "
                    "payload_offset=%d payload_size=%d"
                ),
                image_index,
                element.offset,
                width,
                height,
                payload_offset,
                len(payload),
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
        if images:
            log.debug("Extraction complete: found %d Cell Array image(s)", len(images))
        else:
            top_ids = ", ".join(
                f"({class_id},{element_id})={count}"
                for (class_id, element_id), count in element_histogram.most_common(10)
            )
            log.debug(
                "No Cell Array elements (class=%d id=%d) found in %d parsed elements. "
                "Most frequent element IDs: %s",
                CELL_ARRAY_CLASS_ID,
                CELL_ARRAY_ELEMENT_ID,
                sum(element_histogram.values()),
                top_ids,
            )

    return images


def extract_raw_images(file_path: str | Path) -> list[RawImage]:
    """Extract raw images from a binary CGM file path."""
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
    if log.isEnabledFor(logging.DEBUG):
        log.debug("Writing extracted payloads to %s with stem='%s'", output_dir, stem)

    for image in extract_raw_images(file_path):
        path = image.write(output_dir, stem=stem)
        if log.isEnabledFor(logging.DEBUG):
            log.debug("Wrote payload file: %s (%d bytes)", path, len(image.payload))
        written.append(path)

    if log.isEnabledFor(logging.DEBUG):
        log.debug("Finished writing payload files: %d file(s)", len(written))

    return written


def _split_text_commands(text: str) -> list[str]:
    commands: list[str] = []
    start = 0
    in_quote = False
    quote_char = ""

    for idx, ch in enumerate(text):
        if ch in ('"', "'"):
            if not in_quote:
                in_quote = True
                quote_char = ch
            elif quote_char == ch:
                in_quote = False
                quote_char = ""
            continue

        if ch == ";" and not in_quote:
            part = text[start:idx].strip()
            if part:
                commands.append(part)
            start = idx + 1

    tail = text[start:].strip()
    if tail:
        commands.append(tail)
    return commands


def _extract_text_numbers(statement: str) -> list[float]:
    return [float(item) for item in _TEXT_NUMBER_RE.findall(statement)]


def _extract_hex_payload(statement: str) -> bytes:
    tail = statement
    for marker in ("''", '""'):
        idx = tail.find(marker)
        if idx >= 0:
            tail = tail[idx + len(marker) :]
            break

    tail_chunks = re.findall(r"[0-9A-Fa-f]+", tail)
    if tail_chunks:
        combined = "".join(tail_chunks)
        if len(combined) & 1:
            combined = combined[:-1]
        if combined:
            try:
                return bytes.fromhex(combined)
            except ValueError:
                pass

    words = statement.split()
    # Tile payloads are typically at the end of the statement and may be short.
    tokens: list[str] = []
    for token in words:
        candidate = token.strip().strip("'\",()")
        if len(candidate) < 4 or (len(candidate) & 1):
            continue
        if re.fullmatch(r"[0-9A-Fa-f]+", candidate) is None:
            continue
        tokens.append(candidate)

    if tokens:
        combined = "".join(tokens)
        if len(combined) & 1:
            combined = combined[:-1]
        if combined:
            try:
                return bytes.fromhex(combined)
            except ValueError:
                pass

    runs = _TEXT_HEX_RUN_RE.findall(statement)
    if not runs:
        return b""

    hex_text = "".join(runs)
    if len(hex_text) & 1:
        hex_text = hex_text[:-1]

    if not hex_text:
        return b""

    try:
        return bytes.fromhex(hex_text)
    except ValueError:
        return b""


def _tile_axis_sizes(total: int, count: int) -> list[int]:
    if count <= 0:
        return []
    base = max(1, total // count)
    remainder = max(0, total - (base * count))
    return [base + (1 if idx < remainder else 0) for idx in range(count)]


def _infer_tile_grid(tile_count: int, total_w: int, total_h: int) -> tuple[int, int]:
    if tile_count <= 1:
        return 1, 1

    target_aspect = (total_w / total_h) if total_h > 0 else float(total_w)
    best_cols = 1
    best_rows = tile_count
    best_error = float("inf")

    for rows in range(1, tile_count + 1):
        if tile_count % rows != 0:
            continue
        cols = tile_count // rows
        aspect = cols / rows
        error = abs(aspect - target_aspect)
        if error < best_error:
            best_error = error
            best_cols = cols
            best_rows = rows

    return best_cols, best_rows


_TEXT_TILE_COMMAND_FAMILY: dict[str, str | None] = {
    "BITONALTILE": None,
    "COLORTILE": None,
    "COLOURTILE": None,
    "DIRECTCOLORTILE": None,
    "DIRECTCOLOURTILE": None,
    "INDEXCOLORTILE": "indexed",
    "INDEXCOLOURTILE": "indexed",
    "MONOCHROMETILE": None,
}


def _make_tile_array_record(
    *,
    cols: int,
    rows: int,
    tile_width: int,
    tile_height: int,
    total_width: int,
    total_height: int,
) -> dict[str, object]:
    return {
        "cols": max(1, cols),
        "rows": max(1, rows),
        "tile_width": max(1, tile_width),
        "tile_height": max(1, tile_height),
        "total_width": max(1, total_width),
        "total_height": max(1, total_height),
        "tiles": [],
    }


def _append_tile_record(
    array: dict[str, object],
    *,
    payload: bytes,
    compression: int | None,
    bit_order: int | None,
    orientation: int | None,
    family: str | None,
    local_color_precision: int | None = None,
    cell_representation_mode: int | None = None,
) -> None:
    tiles = array.get("tiles")
    if not isinstance(tiles, list):
        return
    tiles.append(
        {
            "payload": payload,
            "compression": compression,
            "bit_order": bit_order,
            "orientation": orientation,
            "family": family,
            "local_color_precision": local_color_precision,
            "cell_representation_mode": cell_representation_mode,
        }
    )


def _finalize_tile_array_record(array: dict[str, object]) -> dict[str, object] | None:
    tiles = array.get("tiles")
    if not isinstance(tiles, list) or not tiles:
        return None

    cols = _coerce_int(array.get("cols", 1))
    rows = _coerce_int(array.get("rows", 1))
    tile_w = _coerce_int(array.get("tile_width", 1))
    tile_h = _coerce_int(array.get("tile_height", 1))
    total_w = _coerce_int(array.get("total_width", 1))
    total_h = _coerce_int(array.get("total_height", 1))

    tile_pixels = max(1, tile_w * tile_h)
    for tile in tiles:
        if not isinstance(tile, dict):
            continue
        payload = tile.get("payload")
        if not isinstance(payload, (bytes, bytearray)):
            continue
        if isinstance(tile.get("family"), str):
            continue
        local_color_precision = tile.get("local_color_precision")
        # Safe default: one-byte-per-pixel tiles are treated as indexed when
        # metadata suggests palette-friendly precision (or no metadata exists).
        if tile_pixels <= len(payload) < tile_pixels * 2 and (
            local_color_precision is None
            or (isinstance(local_color_precision, int) and local_color_precision <= 8)
        ):
            tile["family"] = "indexed"

    return _make_tile_array_record(
        cols=cols,
        rows=rows,
        tile_width=tile_w,
        tile_height=tile_h,
        total_width=total_w,
        total_height=total_h,
    ) | {"tiles": tiles}


def _parse_binary_tile_arrays(data: bytes) -> list[dict[str, object]]:
    """Synthesize tile-array metadata from binary Cell Array sequences."""
    array = _make_tile_array_record(
        cols=1,
        rows=1,
        tile_width=1,
        tile_height=1,
        total_width=1,
        total_height=1,
    )
    common_total_w: int | None = None
    common_total_h: int | None = None
    tile_count = 0

    for element in iter_elements(data):
        if element.class_id != CELL_ARRAY_CLASS_ID or element.element_id != CELL_ARRAY_ELEMENT_ID:
            continue

        metadata = _parse_cell_array_metadata(element.parameters)
        width = metadata["width"] if isinstance(metadata["width"], int) else None
        height = metadata["height"] if isinstance(metadata["height"], int) else None
        payload_offset = (
            metadata["payload_offset"] if isinstance(metadata["payload_offset"], int) else 0
        )
        payload = (
            element.parameters[payload_offset:] if payload_offset < len(element.parameters) else b""
        )
        if not payload:
            continue

        if width is not None and width > 0:
            common_total_w = width if common_total_w is None else common_total_w
        if height is not None and height > 0:
            common_total_h = height if common_total_h is None else common_total_h

        _append_tile_record(
            array,
            payload=bytes(payload),
            compression=None,
            bit_order=None,
            orientation=None,
            family=None,
            local_color_precision=metadata["local_color_precision"]
            if isinstance(metadata["local_color_precision"], int)
            else None,
            cell_representation_mode=metadata["cell_representation_mode"]
            if isinstance(metadata["cell_representation_mode"], int)
            else None,
        )
        tile_count += 1

    if tile_count == 0:
        return []

    total_w = common_total_w if isinstance(common_total_w, int) and common_total_w > 0 else 1
    total_h = common_total_h if isinstance(common_total_h, int) and common_total_h > 0 else 1
    cols, rows = _infer_tile_grid(tile_count, total_w, total_h)
    tile_w = max(1, total_w // max(1, cols))
    tile_h = max(1, total_h // max(1, rows))
    array["cols"] = cols
    array["rows"] = rows
    array["tile_width"] = tile_w
    array["tile_height"] = tile_h
    array["total_width"] = total_w
    array["total_height"] = total_h

    finalized = _finalize_tile_array_record(array)
    return [finalized] if finalized is not None else []


def _parse_tile_arrays(data: bytes) -> list[dict[str, object]]:
    """Parse tile-array metadata from text commands or synthesized binary hints."""
    text_arrays = _parse_text_tile_arrays(data)
    if text_arrays:
        return text_arrays
    return _parse_binary_tile_arrays(data)


def _parse_text_tile_arrays(data: bytes) -> list[dict[str, object]]:
    try:
        text = data.decode("latin-1")
    except UnicodeDecodeError:
        return []

    arrays: list[dict[str, object]] = []
    current: dict[str, object] | None = None

    for statement in _split_text_commands(text):
        words = statement.split()
        if not words:
            continue

        command = words[0].upper()
        if command == "BEGTILEARRAY":
            numbers = _extract_text_numbers(statement)
            if len(numbers) >= 14:
                cols = max(1, round(numbers[4]))
                rows = max(1, round(numbers[5]))
                tile_w = max(1, round(numbers[6]))
                tile_h = max(1, round(numbers[7]))
                total_w = max(1, round(numbers[-2]))
                total_h = max(1, round(numbers[-1]))
            else:
                cols, rows, tile_w, tile_h, total_w, total_h = 1, 1, 1, 1, 1, 1

            current = _make_tile_array_record(
                cols=cols,
                rows=rows,
                tile_width=tile_w,
                tile_height=tile_h,
                total_width=total_w,
                total_height=total_h,
            )
            continue

        if command in _TEXT_TILE_COMMAND_FAMILY and current is not None:
            payload = _extract_hex_payload(statement)
            if payload:
                numbers = _extract_text_numbers(statement)
                compression = round(numbers[0]) if numbers else None
                local_color_precision = round(numbers[1]) if len(numbers) >= 2 else None
                bit_order = round(numbers[2]) if len(numbers) >= 3 else None
                orientation = round(numbers[3]) if len(numbers) >= 4 else None
                cell_representation_mode = round(numbers[4]) if len(numbers) >= 5 else None
                _append_tile_record(
                    current,
                    payload=payload,
                    compression=compression,
                    bit_order=bit_order,
                    orientation=orientation,
                    family=_TEXT_TILE_COMMAND_FAMILY[command],
                    local_color_precision=local_color_precision,
                    cell_representation_mode=cell_representation_mode,
                )
            continue

        if command == "ENDTILEARRAY" and current is not None:
            finalized = _finalize_tile_array_record(current)
            if finalized is not None:
                arrays.append(finalized)
            current = None

    if current is not None:
        finalized = _finalize_tile_array_record(current)
        if finalized is not None:
            arrays.append(finalized)

    return arrays


def _decode_bitonal_payload_to_image(
    payload: bytes,
    width: int,
    height: int,
    *,
    compression: int | None = None,
    bit_order: int | None = None,
) -> Image.Image | None:
    image, _details = _decode_bitonal_payload_with_details(
        payload,
        width,
        height,
        compression=compression,
        bit_order=bit_order,
    )
    return image


def _decode_tile_payload_to_image(
    payload: bytes,
    width: int,
    height: int,
    *,
    compression: int | None = None,
    bit_order: int | None = None,
    family: str | None = None,
    local_color_precision: int | None = None,
    cell_representation_mode: int | None = None,
    indexed_palette: bytes | None = None,
    color_value_extent: tuple[int, int, int, int, int, int] | None = None,
) -> Image.Image | None:
    """Decode tile payloads across bitonal and direct-color encodings."""
    if Image is None or width <= 0 or height <= 0:
        return None

    total = width * height

    if family == "indexed" and len(payload) >= total:
        image = Image.frombytes("P", (width, height), payload[:total])
        palette = (
            indexed_palette
            if isinstance(indexed_palette, (bytes, bytearray)) and len(indexed_palette) >= 256 * 3
            else bytes(channel for value in range(256) for channel in (value, value, value))
        )
        image.putpalette(palette)
        return image

    # Honor explicit safe color-precision hints first when available.
    if local_color_precision == 32 and len(payload) >= total * 4:
        return Image.frombytes("RGBA", (width, height), payload[: total * 4])

    if local_color_precision == 16:
        scaled_rgb = _scale_direct16_rgb_payload(
            payload,
            total,
            color_value_extent=color_value_extent,
        )
        if scaled_rgb is not None:
            return Image.frombytes("RGB", (width, height), scaled_rgb)

    if local_color_precision == 24 and len(payload) >= total * 3:
        return Image.frombytes("RGB", (width, height), payload[: total * 3])

    if (
        isinstance(local_color_precision, int)
        and local_color_precision <= 8
        and len(payload) >= total
        and family is None
        and (cell_representation_mode in (None, 0, 1))
    ):
        image = Image.frombytes("P", (width, height), payload[:total])
        palette = (
            indexed_palette
            if isinstance(indexed_palette, (bytes, bytearray)) and len(indexed_palette) >= 256 * 3
            else bytes(channel for value in range(256) for channel in (value, value, value))
        )
        image.putpalette(palette)
        return image

    image = _decode_bitonal_payload_to_image(
        payload,
        width,
        height,
        compression=compression,
        bit_order=bit_order,
    )
    if image is not None:
        return image

    if len(payload) >= total * 4:
        return Image.frombytes("RGBA", (width, height), payload[: total * 4])

    if len(payload) >= total * 3:
        return Image.frombytes("RGB", (width, height), payload[: total * 3])

    if len(payload) >= total * 2:
        # Common 16-bit grayscale payloads can be approximated by high bytes.
        gray16 = payload[: total * 2]
        gray8 = bytes(gray16[idx] for idx in range(0, len(gray16), 2))
        return Image.frombytes("L", (width, height), gray8)

    if len(payload) >= total:
        sample = payload[:total]
        unique = set(sample)
        if unique.issubset({0, 1}):
            sample = bytes(0 if value else 255 for value in sample)
        return Image.frombytes("L", (width, height), sample)

    return None


def _decode_bitonal_payload_with_details(
    payload: bytes,
    width: int,
    height: int,
    *,
    compression: int | None = None,
    bit_order: int | None = None,
    preferred_signature: tuple[str, str, bool] | None = None,
    preferred_dimensions: tuple[int, int] | None = None,
) -> tuple[Image.Image | None, dict[str, object]]:
    preferred_score_tolerance = 0.03
    if Image is None or imagecodecs is None or width <= 0 or height <= 0:
        return None, {
            "best_score": None,
            "candidate_count": 0,
            "best_candidate": None,
            "attempts": [],
        }

    best_bits: list[int] | None = None
    best_score = 1e9
    best_candidate: dict[str, object] | None = None
    best_preferred_bits: list[int] | None = None
    best_preferred_score = 1e9
    best_preferred_candidate: dict[str, object] | None = None
    attempts: list[dict[str, object]] = []

    width_candidates = [width]
    if width > 2:
        width_candidates.extend([width - 1, width + 1])
    height_candidates = [height]
    if height > 2:
        height_candidates.extend([height - 1, height + 1])

    size_candidates: list[tuple[int, int]] = []
    seen_sizes: set[tuple[int, int]] = set()
    for cand_w in width_candidates:
        for cand_h in height_candidates:
            key = (cand_w, cand_h)
            if key in seen_sizes or cand_w <= 0 or cand_h <= 0:
                continue
            seen_sizes.add(key)
            size_candidates.append(key)

    payload_candidates = _payload_variants(payload)
    if bit_order == 1 and len(payload_candidates) > 1:
        payload_candidates = [payload_candidates[1], payload_candidates[0]]

    for encoded in payload_candidates:
        encoded_variant = "bit_reversed" if encoded != payload else "as_is"

        # Some profiles store uncompressed packed bits; attempt this path too.
        row_bytes = (width + 7) // 8
        packed_needed = row_bytes * height
        if len(encoded) >= packed_needed:
            packed_bits = _decode_fax_output_to_bitmap(encoded, width, height)
            if packed_bits is not None:
                for invert in (False, True):
                    candidate = [1 - bit for bit in packed_bits] if invert else packed_bits
                    score = _score_bitmap(candidate, width, height)
                    if compression == 0:
                        score -= 0.2
                    signature = ("packed_raw", encoded_variant, invert)
                    attempts.append(
                        {
                            "decoder": "packed_raw",
                            "encoded_variant": encoded_variant,
                            "width": width,
                            "height": height,
                            "invert": invert,
                            "score": round(score, 6),
                        }
                    )
                    if score < best_score:
                        best_score = score
                        best_bits = candidate
                        best_candidate = {
                            "decoder": "packed_raw",
                            "encoded_variant": encoded_variant,
                            "width": width,
                            "height": height,
                            "invert": invert,
                            "score": round(score, 6),
                        }
                    if preferred_signature is not None and signature == preferred_signature:
                        if preferred_dimensions is not None and preferred_dimensions != (
                            width,
                            height,
                        ):
                            continue
                        if score < best_preferred_score:
                            best_preferred_score = score
                            best_preferred_bits = candidate
                            best_preferred_candidate = {
                                "decoder": "packed_raw",
                                "encoded_variant": encoded_variant,
                                "width": width,
                                "height": height,
                                "invert": invert,
                                "score": round(score, 6),
                            }

        for cand_w, cand_h in size_candidates:
            decode_attempts: list[tuple[str, bytes]] = []
            # Respect declared compression when present to avoid pathological brute-force loops.
            allow_fax4 = compression in (None, 2)
            allow_fax3 = compression in (None, 1)

            if allow_fax4:
                try:
                    decode_attempts.append(
                        (
                            "fax4",
                            bytes(
                                imagecodecs.ccittfax4_decode(encoded, height=cand_h, width=cand_w)
                            ),
                        )
                    )
                except (RuntimeError, ValueError):
                    pass

            if allow_fax3:
                for t4options in (0, 2, 4, 6, 1, 3, 5, 7):
                    try:
                        decoded = imagecodecs.ccittfax3_decode(
                            encoded,
                            height=cand_h,
                            width=cand_w,
                            t4options=t4options,
                        )
                        decode_attempts.append((f"fax3:{t4options}", bytes(decoded)))
                    except (RuntimeError, ValueError):
                        continue

            for decoder_name, decoded in decode_attempts:
                bits = _decode_fax_output_to_bitmap(decoded, cand_w, cand_h)
                if bits is None:
                    continue

                if cand_w != width or cand_h != height:
                    tmp_pixels = bytes(0 if bit else 255 for bit in bits)
                    tmp_img = Image.frombytes("L", (cand_w, cand_h), tmp_pixels)
                    tmp_img = tmp_img.resize((width, height), resample=Image.NEAREST)
                    bits = [1 if value == 0 else 0 for value in tmp_img.tobytes()]

                for invert in (False, True):
                    candidate = [1 - bit for bit in bits] if invert else bits
                    score = _score_bitmap(candidate, width, height)
                    if compression in (1, 2):
                        score -= 0.05
                    signature = (decoder_name, encoded_variant, invert)
                    attempts.append(
                        {
                            "decoder": decoder_name,
                            "encoded_variant": encoded_variant,
                            "width": cand_w,
                            "height": cand_h,
                            "invert": invert,
                            "score": round(score, 6),
                        }
                    )
                    if score < best_score:
                        best_score = score
                        best_bits = candidate
                        best_candidate = {
                            "decoder": decoder_name,
                            "encoded_variant": encoded_variant,
                            "width": cand_w,
                            "height": cand_h,
                            "invert": invert,
                            "score": round(score, 6),
                        }
                    if preferred_signature is not None and signature == preferred_signature:
                        if preferred_dimensions is not None and preferred_dimensions != (
                            cand_w,
                            cand_h,
                        ):
                            continue
                        if score < best_preferred_score:
                            best_preferred_score = score
                            best_preferred_bits = candidate
                            best_preferred_candidate = {
                                "decoder": decoder_name,
                                "encoded_variant": encoded_variant,
                                "width": cand_w,
                                "height": cand_h,
                                "invert": invert,
                                "score": round(score, 6),
                            }

    use_preferred = False
    if preferred_signature is not None and best_preferred_bits is not None:
        # Keep global consistency only when it does not materially degrade local decode quality.
        if best_bits is None or best_preferred_score <= (best_score + preferred_score_tolerance):
            use_preferred = True
    selected_bits = best_preferred_bits if use_preferred else best_bits
    selected_score = best_preferred_score if use_preferred else best_score
    selected_candidate = best_preferred_candidate if use_preferred else best_candidate

    if selected_bits is None:
        return None, {
            "best_score": None,
            "candidate_count": len(attempts),
            "best_candidate": None,
            "preferred_signature": preferred_signature,
            "preferred_dimensions": preferred_dimensions,
            "used_preferred_signature": False,
            "attempts": attempts,
        }

    pixels = bytes(0 if bit else 255 for bit in selected_bits)
    return Image.frombytes("L", (width, height), pixels), {
        "best_score": round(selected_score, 6),
        "candidate_count": len(attempts),
        "best_candidate": selected_candidate,
        "preferred_signature": preferred_signature,
        "preferred_dimensions": preferred_dimensions,
        "used_preferred_signature": use_preferred,
        "attempts": attempts,
    }


def _render_raw_image_payload(
    image: RawImage,
    *,
    indexed_palette: bytes | None = None,
    color_value_extent: tuple[int, int, int, int, int, int] | None = None,
) -> Image.Image | None:
    if Image is None or image.width is None or image.height is None:
        return None

    width = image.width
    height = image.height
    if width <= 0 or height <= 0:
        return None

    total = width * height

    if (
        isinstance(image.local_color_precision, int)
        and image.local_color_precision <= 8
        and len(image.payload) >= total
        and (image.cell_representation_mode in (None, 0, 1))
    ):
        indexed = Image.frombytes("P", (width, height), image.payload[:total])
        palette = (
            indexed_palette
            if isinstance(indexed_palette, (bytes, bytearray)) and len(indexed_palette) >= 256 * 3
            else bytes(channel for value in range(256) for channel in (value, value, value))
        )
        indexed.putpalette(palette)
        return indexed

    if image.local_color_precision == 32 and len(image.payload) >= total * 4:
        rgba = image.payload[: total * 4]
        return Image.frombytes("RGBA", (width, height), rgba)

    if image.local_color_precision == 16:
        scaled_rgb = _scale_direct16_rgb_payload(
            image.payload,
            total,
            color_value_extent=color_value_extent,
        )
        if scaled_rgb is not None:
            return Image.frombytes("RGB", (width, height), scaled_rgb)

    if image.local_color_precision == 24 and len(image.payload) >= total * 3:
        rgb = image.payload[: total * 3]
        return Image.frombytes("RGB", (width, height), rgb)

    if len(image.payload) >= total * 4:
        rgba = image.payload[: total * 4]
        return Image.frombytes("RGBA", (width, height), rgba)

    if len(image.payload) >= total * 3:
        rgb = image.payload[: total * 3]
        return Image.frombytes("RGB", (width, height), rgb)

    if len(image.payload) >= total:
        sample = image.payload[:total]
        unique = set(sample)
        if unique.issubset({0, 1}):
            sample = bytes(0 if value else 255 for value in sample)
        return Image.frombytes("L", (width, height), sample)

    row_bytes = (width + 7) // 8
    packed_needed = row_bytes * height
    if len(image.payload) >= packed_needed:
        bits: list[int] = []
        packed = image.payload[:packed_needed]
        for y in range(height):
            row_start = y * row_bytes
            for x in range(width):
                value = packed[row_start + (x // 8)]
                bit = (value >> (7 - (x % 8))) & 1
                bits.append(bit)
        pixels = bytes(0 if bit else 255 for bit in bits)
        return Image.frombytes("L", (width, height), pixels)

    return None


def _coerce_int(value: object, default: int = 1) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return default
        try:
            return int(stripped)
        except ValueError:
            return default
    return default


def _render_first_tile_array(data: bytes) -> Image.Image | None:
    if Image is None:
        return None

    indexed_palette = _indexed_palette_bytes(_extract_color_table(data))
    color_value_extent = _extract_color_value_extent(data)

    for array in _parse_tile_arrays(data):
        cols = _coerce_int(array.get("cols", 1))
        rows = _coerce_int(array.get("rows", 1))
        tile_w_nominal = _coerce_int(array.get("tile_width", 1))
        tile_h_nominal = _coerce_int(array.get("tile_height", 1))
        total_w = _coerce_int(array.get("total_width", 1))
        total_h = _coerce_int(array.get("total_height", 1))
        tiles = array.get("tiles", [])
        if not isinstance(tiles, list) or not tiles:
            continue

        canvas = Image.new("L", (total_w, total_h), color=255)
        pasted_any = False

        for tile_index, tile_payload in enumerate(tiles[: rows * cols]):
            if not isinstance(tile_payload, dict):
                continue

            payload = tile_payload.get("payload")
            compression = tile_payload.get("compression")
            bit_order = tile_payload.get("bit_order")
            if not isinstance(payload, (bytes, bytearray)):
                continue

            tile_img = _decode_tile_payload_to_image(
                bytes(payload),
                tile_w_nominal,
                tile_h_nominal,
                compression=compression if isinstance(compression, int) else None,
                bit_order=bit_order if isinstance(bit_order, int) else None,
                family=tile_payload.get("family")
                if isinstance(tile_payload.get("family"), str)
                else None,
                local_color_precision=tile_payload.get("local_color_precision")
                if isinstance(tile_payload.get("local_color_precision"), int)
                else None,
                cell_representation_mode=tile_payload.get("cell_representation_mode")
                if isinstance(tile_payload.get("cell_representation_mode"), int)
                else None,
                indexed_palette=indexed_palette,
                color_value_extent=color_value_extent,
            )
            if tile_img is None:
                continue

            row = tile_index // cols
            col = tile_index % cols
            x = col * tile_w_nominal
            y = row * tile_h_nominal
            if x >= total_w or y >= total_h:
                continue

            paste_w = min(tile_img.width, total_w - x)
            paste_h = min(tile_img.height, total_h - y)
            if paste_w <= 0 or paste_h <= 0:
                continue

            tile_for_canvas = tile_img
            if tile_img.size != (paste_w, paste_h):
                tile_for_canvas = tile_img.crop((0, 0, paste_w, paste_h))
            canvas.paste(tile_for_canvas, (x, y))
            pasted_any = True

        if pasted_any:
            return canvas

    return None


def _render_first_tile_array_overlays(
    data: bytes,
) -> tuple[list[tuple[str, float, float, float, float]], int, int] | None:
    """Decode first available tile array and return per-tile SVG overlays."""
    indexed_palette = _indexed_palette_bytes(_extract_color_table(data))
    color_value_extent = _extract_color_value_extent(data)

    for array in _parse_tile_arrays(data):
        cols = _coerce_int(array.get("cols", 1))
        rows = _coerce_int(array.get("rows", 1))
        tile_w_nominal = _coerce_int(array.get("tile_width", 1))
        tile_h_nominal = _coerce_int(array.get("tile_height", 1))
        total_w = _coerce_int(array.get("total_width", 1))
        total_h = _coerce_int(array.get("total_height", 1))
        tiles = array.get("tiles", [])
        if not isinstance(tiles, list) or not tiles:
            continue

        overlays: list[tuple[str, float, float, float, float]] = []

        for tile_index, tile_payload in enumerate(tiles[: rows * cols]):
            if not isinstance(tile_payload, dict):
                continue

            payload = tile_payload.get("payload")
            compression = tile_payload.get("compression")
            bit_order = tile_payload.get("bit_order")
            if not isinstance(payload, (bytes, bytearray)):
                continue

            tile_img = _decode_tile_payload_to_image(
                bytes(payload),
                tile_w_nominal,
                tile_h_nominal,
                compression=compression if isinstance(compression, int) else None,
                bit_order=bit_order if isinstance(bit_order, int) else None,
                family=tile_payload.get("family")
                if isinstance(tile_payload.get("family"), str)
                else None,
                local_color_precision=tile_payload.get("local_color_precision")
                if isinstance(tile_payload.get("local_color_precision"), int)
                else None,
                cell_representation_mode=tile_payload.get("cell_representation_mode")
                if isinstance(tile_payload.get("cell_representation_mode"), int)
                else None,
                indexed_palette=indexed_palette,
                color_value_extent=color_value_extent,
            )
            if tile_img is None:
                continue

            row = tile_index // cols
            col = tile_index % cols
            x = col * tile_w_nominal
            y = row * tile_h_nominal
            if x >= total_w or y >= total_h:
                continue

            paste_w = min(tile_img.width, total_w - x)
            paste_h = min(tile_img.height, total_h - y)
            if paste_w <= 0 or paste_h <= 0:
                continue

            tile_for_svg = tile_img
            if tile_img.size != (paste_w, paste_h):
                tile_for_svg = tile_img.crop((0, 0, paste_w, paste_h))

            href = _image_to_png_data_uri(tile_for_svg)
            if href is None:
                continue

            overlays.append((href, float(x), float(y), float(paste_w), float(paste_h)))

        if overlays:
            return overlays, total_w, total_h

    return None


def _render_raster_background_data_uri(data: bytes) -> tuple[str, int, int] | None:
    tiled = _render_first_tile_array(data)
    if tiled is not None:
        href = _image_to_png_data_uri(tiled)
        if href is not None:
            return href, tiled.width, tiled.height

    indexed_palette = _indexed_palette_bytes(_extract_color_table(data))
    color_value_extent = _extract_color_value_extent(data)

    for image in extract_raw_images_from_bytes(data):
        rendered = _render_raw_image_payload(
            image,
            indexed_palette=indexed_palette,
            color_value_extent=color_value_extent,
        )
        if rendered is None:
            continue
        href = _image_to_png_data_uri(rendered)
        if href is not None:
            return href, rendered.width, rendered.height

    return None


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
    svg_path.write_text(extract_vector_svg_from_bytes(raw), encoding="utf-8")
    written.append(svg_path)

    if debug_report:
        arrays_report: list[dict[str, object]] = []
        for idx, array in enumerate(_parse_tile_arrays(raw)):
            tiles = array.get("tiles", [])
            arrays_report.append(
                {
                    "array_index": idx,
                    "cols": _coerce_int(array.get("cols", 1)),
                    "rows": _coerce_int(array.get("rows", 1)),
                    "tile_width": _coerce_int(array.get("tile_width", 1)),
                    "tile_height": _coerce_int(array.get("tile_height", 1)),
                    "total_width": _coerce_int(array.get("total_width", 1)),
                    "total_height": _coerce_int(array.get("total_height", 1)),
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
