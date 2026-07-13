"""High-level CGM image extraction helpers."""

from __future__ import annotations

__author__ = "Kestin Goforth"
__copyright__ = "Copyright 2026"
__license__ = "BSD-3-Clause"

import html
import json
import logging
import math
import struct
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from .parser import CELL_ARRAY_CLASS_ID, CELL_ARRAY_ELEMENT_ID, iter_elements
from .types import RawImage

log = logging.getLogger("cgm.extract")


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


@dataclass(slots=True)
class _SvgStyle:
    stroke_color: str = "#000000"
    stroke_width: float = 1.0
    font_size: float = 10.0


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
        if abs(x) > 1_000_000 or abs(y) > 1_000_000:
            continue
        points.append((x, y))
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
            if abs(x) > 10_000 or abs(y) > 10_000:
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


def _palette_index_to_hex(color_index: int) -> str:
    palette = {
        0: "#000000",
        1: "#ffffff",
        2: "#000000",
        3: "#2f2f2f",
        4: "#404040",
        5: "#606060",
        6: "#808080",
        7: "#a0a0a0",
    }
    return palette.get(color_index, "#000000")


def _format_points(points: list[tuple[float, float]], *, min_y: float, max_y: float) -> str:
    mapped = (f"{x:.3f},{(max_y - (y - min_y)):.3f}" for x, y in points)
    return " ".join(mapped)


def extract_vector_svg_from_bytes(data: bytes) -> str:
    """Convert vector-like CGM primitives from bytes into a best-effort SVG string."""
    style = _SvgStyle()
    text_items: list[tuple[float, float, str, str, float]] = []
    polyline_items: list[tuple[list[tuple[float, float]], str, float]] = []
    polygon_items: list[tuple[list[tuple[float, float]], str, float]] = []
    gdp_parameters: list[bytes] = []

    for element in iter_elements(data):
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

    if all_points:
        min_x = min(point[0] for point in all_points)
        max_x = max(point[0] for point in all_points)
        min_y = min(point[1] for point in all_points)
        max_y = max(point[1] for point in all_points)
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
        '  <g fill="none" stroke-linecap="round" stroke-linejoin="round">',
    ]

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

    raw_images = extract_raw_images_from_bytes(data)
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
        "raw_images": raw_image_items,
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

        width, height, payload_offset = _parse_cell_array_hints(element.parameters)
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
