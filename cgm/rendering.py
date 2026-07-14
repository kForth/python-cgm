"""CGM SVG composition and rendered-image export helpers backed by Pillow and imagecodecs."""

from __future__ import annotations

import base64
import html
import io
import json
import logging
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imagecodecs
from PIL import Image

from .common import (
    _decode_cgm_text,
    _decode_point_pairs_exact,
    _decode_point_pairs_heuristic,
    _decode_restricted_text,
    _parse_f32_be,
)
from .parser import iter_elements
from .raster import _analyze_element29_payload
from .raw_images import extract_raw_images_from_bytes

log = logging.getLogger("cgm.rendering")

_TEXT_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
_TEXT_HEX_RUN_RE = re.compile(r"[0-9A-Fa-f]{32,}")


def _coerce_int(value: object, default: int) -> int:
    return value if isinstance(value, int) else default


def _reverse_bits_in_byte(value: int) -> int:
    value = ((value & 0xF0) >> 4) | ((value & 0x0F) << 4)
    value = ((value & 0xCC) >> 2) | ((value & 0x33) << 2)
    value = ((value & 0xAA) >> 1) | ((value & 0x55) << 1)
    return value


def _payload_variants(payload: bytes) -> list[bytes]:
    variants = [payload]
    reversed_payload = payload[::-1]
    if reversed_payload not in variants:
        variants.append(reversed_payload)
    reversed_bits = bytes(_reverse_bits_in_byte(byte) for byte in payload)
    if reversed_bits not in variants:
        variants.append(reversed_bits)
    reversed_bits_reversed = reversed_bits[::-1]
    if reversed_bits_reversed not in variants:
        variants.append(reversed_bits_reversed)
    return variants


@dataclass
class _SvgStyle:
    stroke_color: str = "#000000"
    stroke_width: float = 1.0
    font_size: float = 12.0


def _palette_index_to_hex(color_index: int) -> str:
    palette = {
        0: "#ffffff",
        1: "#000000",
        2: "#ff0000",
        3: "#00ff00",
        4: "#0000ff",
        5: "#00ffff",
        6: "#ff00ff",
        7: "#ffff00",
    }
    if color_index in palette:
        return palette[color_index]
    channel = max(0, min(255, int(color_index)))
    return f"#{channel:02x}{channel:02x}{channel:02x}"


def _format_points(points: list[tuple[float, float]], *, min_y: float, max_y: float) -> str:
    return " ".join(f"{x:.3f},{(max_y - (y - min_y)):.3f}" for x, y in points)


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

            current = {
                "cols": cols,
                "rows": rows,
                "tile_width": tile_w,
                "tile_height": tile_h,
                "total_width": total_w,
                "total_height": total_h,
                "tiles": [],
            }
            continue

        if command == "BITONALTILE" and current is not None:
            payload = _extract_hex_payload(statement)
            if payload:
                numbers = _extract_text_numbers(statement)
                compression = round(numbers[0]) if numbers else None
                bit_order = round(numbers[2]) if len(numbers) >= 3 else None
                orientation = round(numbers[3]) if len(numbers) >= 4 else None
                cast_tiles = current["tiles"]
                if isinstance(cast_tiles, list):
                    cast_tiles.append(
                        {
                            "payload": payload,
                            "compression": compression,
                            "bit_order": bit_order,
                            "orientation": orientation,
                        }
                    )
            continue

        if command == "ENDTILEARRAY" and current is not None:
            arrays.append(current)
            current = None

    if current is not None:
        arrays.append(current)

    return arrays


def _decode_bitonal_payload_to_image(
    payload: bytes,
    width: int,
    height: int,
    *,
    compression: int | None = None,
    bit_order: int | None = None,
) -> Any | None:
    image, _details = _decode_bitonal_payload_with_details(
        payload,
        width,
        height,
        compression=compression,
        bit_order=bit_order,
    )
    return image


def _decode_bitonal_payload_with_details(
    payload: bytes,
    width: int,
    height: int,
    *,
    compression: int | None = None,
    bit_order: int | None = None,
    preferred_signature: tuple[str, str, bool] | None = None,
    preferred_dimensions: tuple[int, int] | None = None,
) -> tuple[Any | None, dict[str, object]]:
    _ = compression, bit_order
    if width <= 0 or height <= 0:
        return None, {
            "best_score": None,
            "candidate_count": 0,
            "best_candidate": None,
            "attempts": [],
        }

    total = width * height
    if total <= 0:
        return None, {
            "best_score": None,
            "candidate_count": 0,
            "best_candidate": None,
            "attempts": [],
        }

    sample = payload[:total]
    if len(sample) < total:
        sample = sample.ljust(total, b"\xff")
    if sample and set(sample).issubset({0, 1}):
        sample = bytes(0 if value else 255 for value in sample)

    image = Image.frombytes("L", (width, height), sample)
    return image, {
        "best_score": 0.0,
        "candidate_count": 1,
        "best_candidate": {
            "decoder": "raw",
            "encoded_variant": "as_is",
            "width": width,
            "height": height,
            "invert": False,
            "score": 0.0,
        },
        "preferred_signature": preferred_signature,
        "preferred_dimensions": preferred_dimensions,
        "used_preferred_signature": False,
        "attempts": [
            {
                "decoder": "raw",
                "encoded_variant": "as_is",
                "width": width,
                "height": height,
                "invert": False,
                "score": 0.0,
            }
        ],
    }


def _decode_fax_output_to_bitmap(output: bytes, width: int, height: int) -> list[int] | None:
    total = width * height
    if total <= 0:
        return None

    if len(output) >= total:
        sample = output[:total]
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
    total = len(bits)
    if total == 0:
        return 1e9

    black = sum(bits)
    black_ratio = black / total
    dominant_ratio = max(black_ratio, 1.0 - black_ratio)
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
    score = 1.0 - transition_ratio
    if transition_ratio < 0.2:
        score += 0.8
    elif transition_ratio < 0.35:
        score += 0.3

    if max(h_density, v_density) < 0.01:
        score += 0.3
    return score


def score_bitmap(bits: list[int], width: int, height: int) -> float:
    return _score_bitmap(bits, width, height)


def _bitmap_to_png_data_uri(bits: list[int], width: int, height: int) -> str | None:
    if Image is None:
        return None
    if len(bits) != width * height:
        return None

    pixels = bytes(0 if bit else 255 for bit in bits)
    image = Image.frombytes("L", (width, height), pixels)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def _image_to_png_data_uri(image: Any | None) -> str | None:
    if image is None:
        return None

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def _decode_element29_binary_raster(parameters: bytes) -> tuple[str, int, int] | None:
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
            for codec_name in ("ccittfax3", "ccittfax4", "ccittfax2"):
                decode = getattr(imagecodecs, f"{codec_name}_decode", None)
                if decode is None:
                    continue
                for payload_variant in (payload, payload[::-1]):
                    try:
                        decoded = decode(payload_variant, shape=(height, width))
                    except (AttributeError, RuntimeError, TypeError, ValueError):
                        continue
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
                            best_bits = candidate
                            best_size = (width, height)
                            best_black_ratio = black_ratio

    if best_bits is None or best_size is None:
        return None

    width, height = best_size
    data_uri = _bitmap_to_png_data_uri(best_bits, width, height)
    if data_uri is None:
        return None
    return data_uri, width, height


def extract_vector_svg_from_bytes(data: bytes) -> str:
    style = _SvgStyle()
    text_items: list[tuple[float, float, str, str, float]] = []
    restricted_text_items: list[tuple[float, float, float, float, str, str, float]] = []
    polyline_items: list[tuple[list[tuple[float, float]], str, float]] = []
    polygon_items: list[tuple[list[tuple[float, float]], str, float]] = []
    gdp_parameters: list[bytes] = []
    unsupported_class4: Counter[int] = Counter()
    element29_payloads: list[bytes] = []
    element29_raster_rendered = False
    vdc_extent: tuple[float, float, float, float] | None = None
    raster_background = _render_raster_background_data_uri(data)

    for element in iter_elements(data):
        if element.class_id == 2 and element.element_id == 6 and len(element.parameters) >= 8:
            x1 = int.from_bytes(element.parameters[0:2], "big", signed=False)
            y1 = int.from_bytes(element.parameters[2:4], "big", signed=False)
            x2 = int.from_bytes(element.parameters[4:6], "big", signed=False)
            y2 = int.from_bytes(element.parameters[6:8], "big", signed=False)
            if x1 != x2 and y1 != y2:
                vdc_extent = (
                    float(min(x1, x2)),
                    float(min(y1, y2)),
                    float(max(x1, x2)),
                    float(max(y1, y2)),
                )
            continue

        if element.class_id == 5:
            if element.element_id == 3 and len(element.parameters) >= 4:
                width = _parse_f32_be(element.parameters[:4])
                if width is not None and width > 0:
                    style.stroke_width = max(0.25, width)
            elif element.element_id == 4 and element.parameters:
                style.stroke_color = _palette_index_to_hex(element.parameters[0])
            elif element.element_id == 15 and len(element.parameters) >= 4:
                char_height = _parse_f32_be(element.parameters[:4])
                if char_height is not None and char_height > 0:
                    style.font_size = max(6.0, char_height)
            continue

        if element.class_id != 4:
            continue

        if element.element_id == 1:
            points = _decode_point_pairs_exact(element.parameters)
            if len(points) >= 2:
                polyline_items.append((points, style.stroke_color, style.stroke_width))
        elif element.element_id == 7:
            points = _decode_point_pairs_exact(element.parameters)
            if len(points) >= 3:
                polygon_items.append((points, style.stroke_color, style.stroke_width))
        elif element.element_id == 5 and len(element.parameters) >= 8:
            x = _parse_f32_be(element.parameters[0:4])
            y = _parse_f32_be(element.parameters[4:8])
            text = _decode_cgm_text(element.parameters[8:])
            if x is not None and y is not None and text:
                text_items.append((x, y, text, style.stroke_color, style.font_size))
        elif element.element_id == 26:
            gdp_parameters.append(element.parameters)
        elif element.element_id == 29:
            restricted = _decode_restricted_text(element.parameters)
            if restricted is not None:
                anchor_x, anchor_y, box_w, box_h, text = restricted
                restricted_text_items.append(
                    (anchor_x, anchor_y, box_w, box_h, text, style.stroke_color, style.font_size)
                )
            else:
                element29_payloads.append(element.parameters)
                unsupported_class4[element.element_id] += 1
        else:
            unsupported_class4[element.element_id] += 1

    if not polyline_items and not polygon_items:
        for parameters in gdp_parameters:
            points = _decode_point_pairs_heuristic(parameters)
            if len(points) >= 2:
                polyline_items.append((points, style.stroke_color, style.stroke_width))

    all_points: list[tuple[float, float]] = []
    for points, _, _ in polyline_items:
        all_points.extend(points)
    for points, _, _ in polygon_items:
        all_points.extend(points)
    for x, y, _, _, _ in text_items:
        all_points.append((x, y))
    for x, y, box_w, box_h, _, _, _ in restricted_text_items:
        all_points.append((x, y))
        all_points.append((x + box_w, y + box_h))

    if all_points:
        min_x = min(point[0] for point in all_points)
        max_x = max(point[0] for point in all_points)
        min_y = min(point[1] for point in all_points)
        max_y = max(point[1] for point in all_points)
    elif vdc_extent is not None:
        min_x, min_y, max_x, max_y = vdc_extent
    elif raster_background is not None:
        _href, raster_w, raster_h = raster_background
        min_x, min_y, max_x, max_y = 0.0, 0.0, float(raster_w), float(raster_h)
    else:
        min_x, min_y, max_x, max_y = 0.0, 0.0, 100.0, 100.0

    width = max(1.0, max_x - min_x)
    height = max(1.0, max_y - min_y)

    svg_lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'viewBox="{min_x:.3f} {min_y:.3f} {width:.3f} {height:.3f}">'
        ),
    ]

    if raster_background is not None:
        href, _raster_w, _raster_h = raster_background
        svg_lines.append(
            f'  <image x="{min_x:.3f}" y="{min_y:.3f}" width="{width:.3f}" '
            f'height="{height:.3f}" preserveAspectRatio="xMidYMid meet" '
            f'image-rendering="pixelated" href="{href}" />'
        )

    svg_lines.append('  <g fill="none" stroke-linecap="round" stroke-linejoin="round">')

    for points, stroke_color, stroke_width in polyline_items:
        point_str = _format_points(points, min_y=min_y, max_y=max_y)
        svg_lines.append(
            f'    <polyline points="{point_str}" stroke="{stroke_color}" '
            f'stroke-width="{stroke_width:.3f}" />'
        )

    for points, stroke_color, stroke_width in polygon_items:
        point_str = _format_points(points, min_y=min_y, max_y=max_y)
        svg_lines.append(
            f'    <polygon points="{point_str}" stroke="{stroke_color}" '
            f'stroke-width="{stroke_width:.3f}" fill="none" />'
        )

    svg_lines.append("  </g>")

    for x, y, text, stroke_color, font_size in text_items:
        mapped_y = max_y - (y - min_y)
        escaped = html.escape(text)
        svg_lines.append(
            f'  <text x="{x:.3f}" y="{mapped_y:.3f}" '
            f'fill="{stroke_color}" font-size="{font_size:.3f}" '
            f'font-family="sans-serif">{escaped}</text>'
        )

    for x, y, box_w, box_h, text, stroke_color, font_size in restricted_text_items:
        mapped_y = max_y - (y - min_y)
        box_width = max(1.0, abs(box_w))
        capped_font = max(4.0, min(font_size, abs(box_h)))
        escaped = html.escape(text)
        svg_lines.append(
            f'  <text x="{x:.3f}" y="{mapped_y:.3f}" '
            f'fill="{stroke_color}" font-size="{capped_font:.3f}" '
            f'font-family="sans-serif" textLength="{box_width:.3f}" '
            f'lengthAdjust="spacingAndGlyphs">{escaped}</text>'
        )

    if (
        not polyline_items
        and not polygon_items
        and not text_items
        and not restricted_text_items
        and raster_background is None
        and element29_payloads
    ):
        decoded = _decode_element29_binary_raster(element29_payloads[0])
        if decoded is not None:
            href, raster_width, raster_height = decoded
            element29_raster_rendered = True
            svg_lines.append(
                f'  <image x="{min_x:.3f}" y="{min_y:.3f}" width="{width:.3f}" '
                f'height="{height:.3f}" preserveAspectRatio="xMidYMid meet" '
                f'image-rendering="pixelated" href="{href}" />'
            )
            if log.isEnabledFor(logging.DEBUG):
                log.debug(
                    "Rendered class-4/id-29 payload as raster image (%dx%d)",
                    raster_width,
                    raster_height,
                )

    if (
        not polyline_items
        and not polygon_items
        and not text_items
        and not restricted_text_items
        and raster_background is None
        and not element29_raster_rendered
        and unsupported_class4
    ):
        unsupported_items = list(unsupported_class4.most_common(4))
        unsupported_summary = ", ".join(
            f"id {element_id} x{count}" for element_id, count in unsupported_items
        )
        diagnostics = ""
        if element29_payloads:
            analysis = _analyze_element29_payload(element29_payloads[0])
            ascii_ratio = analysis.get("ascii_ratio", 0.0)
            entropy_value = analysis.get("entropy_bits_per_byte", 0.0)
            leading_ff_value = analysis.get("leading_ff_run", 0)
            ratio = float(ascii_ratio) * 100 if isinstance(ascii_ratio, (int, float)) else 0.0
            entropy = float(entropy_value) if isinstance(entropy_value, (int, float)) else 0.0
            leading_ff = int(leading_ff_value) if isinstance(leading_ff_value, int) else 0
            diagnostics = (
                f"Element 29 payload looks binary (printable ASCII: {ratio:.1f}%, "
                f"entropy: {entropy:.2f}, leading 0xFF run: {leading_ff}). "
                "Expected Restricted Text fields were not detected."
            )
        fallback_font = max(10.0, min(24.0, height * 0.04))
        line_gap = fallback_font * 1.3
        left = min_x + width * 0.04
        top = min_y + height * 0.15

        svg_lines.append(
            f'  <rect x="{min_x:.3f}" y="{min_y:.3f}" width="{width:.3f}" '
            f'height="{height:.3f}" fill="#f7f7f7" stroke="#666" stroke-width="1" />'
        )
        svg_lines.append(
            f'  <text x="{left:.3f}" y="{top:.3f}" fill="#111" font-size="{fallback_font:.3f}" '
            'font-family="sans-serif">CGM contains unsupported drawing primitives.</text>'
        )
        svg_lines.append(
            f'  <text x="{left:.3f}" y="{(top + line_gap):.3f}" fill="#333" '
            f'font-size="{(fallback_font * 0.9):.3f}" font-family="sans-serif">'
            f"Detected: {html.escape(unsupported_summary)}</text>"
        )

        if diagnostics:
            svg_lines.append(
                f'  <text x="{left:.3f}" y="{(top + line_gap * 2):.3f}" fill="#333" '
                f'font-size="{(fallback_font * 0.75):.3f}" font-family="sans-serif">'
                f"{html.escape(diagnostics)}</text>"
            )

        if log.isEnabledFor(logging.DEBUG):
            log.debug("SVG fallback used; unsupported class-4 elements: %s", unsupported_summary)
            if diagnostics:
                log.debug("Element 29 diagnostics: %s", diagnostics)

    svg_lines.append("</svg>")
    return "\n".join(svg_lines) + "\n"


def extract_vector_svg(file_path: str | Path) -> str:
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
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{stem}_0000.svg"
    svg = extract_vector_svg(file_path)
    target.write_text(svg, encoding="utf-8")
    if log.isEnabledFor(logging.DEBUG):
        log.debug("Wrote vector SVG: %s", target)
    return target


def _render_raw_image_payload(image: Any) -> Any | None:
    if image.width is None or image.height is None:
        return None

    width = image.width
    height = image.height
    if width <= 0 or height <= 0:
        return None

    total = width * height
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


def _render_first_text_tile_array(data: bytes) -> Any | None:
    for array in _parse_text_tile_arrays(data):
        cols = _coerce_int(array.get("cols"), 1)
        rows = _coerce_int(array.get("rows"), 1)
        tile_w_nominal = _coerce_int(array.get("tile_width"), 1)
        tile_h_nominal = _coerce_int(array.get("tile_height"), 1)
        total_w = _coerce_int(array.get("total_width"), 1)
        total_h = _coerce_int(array.get("total_height"), 1)
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

            tile_img = _decode_bitonal_payload_to_image(
                bytes(payload),
                tile_w_nominal,
                tile_h_nominal,
                compression=compression if isinstance(compression, int) else None,
                bit_order=bit_order if isinstance(bit_order, int) else None,
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


def _render_raster_background_data_uri(data: bytes) -> tuple[str, int, int] | None:
    tiled = _render_first_text_tile_array(data)
    if tiled is not None:
        href = _image_to_png_data_uri(tiled)
        if href is not None:
            return href, tiled.width, tiled.height

    for image in extract_raw_images_from_bytes(data):
        rendered = _render_raw_image_payload(image)
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
        for idx, array in enumerate(_parse_text_tile_arrays(raw)):
            tiles = array.get("tiles", [])
            arrays_report.append(
                {
                    "array_index": idx,
                    "cols": _coerce_int(array.get("cols"), 1),
                    "rows": _coerce_int(array.get("rows"), 1),
                    "tile_width": _coerce_int(array.get("tile_width"), 1),
                    "tile_height": _coerce_int(array.get("tile_height"), 1),
                    "total_width": _coerce_int(array.get("total_width"), 1),
                    "total_height": _coerce_int(array.get("total_height"), 1),
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
