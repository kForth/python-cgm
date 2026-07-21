"""Vector-to-SVG extraction functions."""

# ruff: noqa: I001

from __future__ import annotations

import base64
from dataclasses import dataclass
import html
import io
import logging
from pathlib import Path
from typing import Any

from cgm.parser import iter_elements
from cgm.extract.core import (
    coerce_int,
    decode_cgm_text,
    decode_circle_data,
    decode_clip_indicator,
    decode_ellipse_data,
    decode_gdp_polyline_points,
    decode_pairwise_segments,
    decode_point_pairs_from_profile,
    decode_rectangle_corners,
    decode_restricted_text,
    decode_transparency_mode,
    decode_vdc_extent,
    decode_xy,
    extract_color_table,
    extract_color_value_extent,
    extract_descriptor_profile,
    indexed_palette_bytes,
    palette_index_to_hex,
    parse_f32_be,
    render_raw_image_payload,
)
from cgm.extract.tiles import (
    _decode_tile_payload_to_image,
    _infer_tile_grid,
    _parse_tile_arrays,
    _render_first_tile_array,
)

log = logging.getLogger("cgm.extract")

_ID29_RASTER_MODELS: dict[tuple[int, int, int, int], tuple[tuple[int, int], ...]] = {
    # Corpus-derived profile from sample fixtures:
    # prefix (compression, row_padding, bit_order, orientation)
    # ordered decode attempts (compression, bit_order)
    (2, 0, 1, 0): ((2, 1), (2, 0), (1, 0), (1, 1)),
    # Learned from remaining misses in the expanded test_files corpus.
    (7, 0, 8, 6): ((1, 0), (1, 1), (2, 1), (2, 0), (0, 0), (0, 1)),
}


@dataclass(slots=True)
class _SvgStyle:
    stroke_color: str = "#000000"
    stroke_width: float = 1.0
    font_size: float = 10.0
    marker_size: float = 2.0


def _format_points(
    points: list[tuple[float, float]],
    *,
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


def _image_to_png_data_uri(image: Any | None) -> str | None:
    if image is None:
        return None

    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    payload = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def _derive_extent_pixel_size(
    vdc_extent: tuple[float, float, float, float] | None,
) -> tuple[int, int] | None:
    if vdc_extent is None:
        return None

    min_x, min_y, max_x, max_y = vdc_extent
    width = round(abs(max_x - min_x))
    height = round(abs(max_y - min_y))
    if width <= 0 or height <= 0:
        return None
    return width, height


def _parse_element29_raster_prefix(
    parameters: bytes,
) -> tuple[int, int, int, int, bytes] | None:
    """Parse class-4/id-29 raster prefix layout.

    Layout: ``u16 compression``, ``u16 row_padding``, ``u16 bit_order``,
    ``u8 orientation``, then encoded raster payload bytes.
    """
    if len(parameters) <= 7:
        return None

    compression = int.from_bytes(parameters[0:2], "big", signed=False)
    row_padding = int.from_bytes(parameters[2:4], "big", signed=False)
    bit_order = int.from_bytes(parameters[4:6], "big", signed=False)
    orientation = parameters[6]
    payload = parameters[7:]
    if not payload:
        return None
    return compression, row_padding, bit_order, orientation, payload


def _decode_element29_payload_with_fallbacks(
    payload: bytes,
    *,
    width: int,
    height: int,
    compression: int,
    row_padding: int,
    bit_order: int,
    orientation: int,
) -> Any | None:
    """Decode id-29 payload using deterministic corpus model candidates.

    Each known prefix maps to a fixed decode attempt sequence and the first
    candidate that decodes successfully is selected.
    """
    model_key = (compression, row_padding, bit_order, orientation)
    candidates = _ID29_RASTER_MODELS.get(model_key)
    if candidates is None:
        return None

    for compression_value, bit_order_value in candidates:
        image = _decode_tile_payload_to_image(
            payload,
            width,
            height,
            compression=compression_value,
            bit_order=bit_order_value,
            row_padding=row_padding,
            orientation=orientation,
        )
        if image is not None:
            return image

    return None


def _collect_element29_raster_payloads(
    data: bytes,
) -> tuple[tuple[int, int] | None, list[tuple[int, int, int, int, bytes]]]:
    vdc_extent: tuple[float, float, float, float] | None = None
    payloads: list[tuple[int, int, int, int, bytes]] = []

    for element in iter_elements(data):
        if element.class_id == 2 and element.element_id == 6:
            decoded_extent = decode_vdc_extent(element.parameters)
            if decoded_extent is not None:
                vdc_extent = decoded_extent
            continue

        if element.class_id == 4 and element.element_id == 29:
            if decode_restricted_text(element.parameters) is not None:
                continue
            parsed = _parse_element29_raster_prefix(element.parameters)
            if parsed is not None:
                payloads.append(parsed)

    return _derive_extent_pixel_size(vdc_extent), payloads


def _render_element29_overlays(
    data: bytes,
) -> tuple[list[tuple[str, float, float, float, float]], int, int] | None:
    size_hint, payloads = _collect_element29_raster_payloads(data)
    if size_hint is None or len(payloads) < 2:
        return None

    total_w, total_h = size_hint
    cols, rows = _infer_tile_grid(len(payloads), total_w, total_h)
    tile_w = max(1, total_w // max(1, cols))
    tile_h = max(1, total_h // max(1, rows))
    overlays: list[tuple[str, float, float, float, float]] = []

    for tile_index, (compression, row_padding, bit_order, orientation, payload) in enumerate(
        payloads
    ):
        tile_image = _decode_element29_payload_with_fallbacks(
            payload,
            width=tile_w,
            height=tile_h,
            compression=compression,
            row_padding=row_padding,
            bit_order=bit_order,
            orientation=orientation,
        )
        if tile_image is None:
            return None

        href = _image_to_png_data_uri(tile_image)
        if href is None:
            return None

        row = tile_index // cols
        col = tile_index % cols
        x = float(col * tile_w)
        y = float(row * tile_h)
        overlays.append((href, x, y, float(tile_w), float(tile_h)))

    return overlays, total_w, total_h


def _render_first_element29_raster(
    data: bytes,
) -> Any | None:
    size_hint, payloads = _collect_element29_raster_payloads(data)
    if size_hint is None:
        return None

    width, height = size_hint
    for compression, row_padding, bit_order, orientation, payload in payloads:
        image = _decode_element29_payload_with_fallbacks(
            payload,
            width=width,
            height=height,
            compression=compression,
            row_padding=row_padding,
            bit_order=bit_order,
            orientation=orientation,
        )
        if image is not None:
            return image

    return None


def _render_first_tile_array_overlays(
    data: bytes,
) -> tuple[list[tuple[str, float, float, float, float]], int, int] | None:
    """Decode first available tile array and return per-tile SVG overlays."""
    indexed_palette = indexed_palette_bytes(extract_color_table(data))
    color_value_extent = extract_color_value_extent(data)

    for array in _parse_tile_arrays(data):
        cols = coerce_int(array.get("cols", 1))
        rows = coerce_int(array.get("rows", 1))
        tile_w_nominal = coerce_int(array.get("tile_width", 1))
        tile_h_nominal = coerce_int(array.get("tile_height", 1))
        total_w = coerce_int(array.get("total_width", 1))
        total_h = coerce_int(array.get("total_height", 1))
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
                row_padding=tile_payload.get("row_padding")
                if isinstance(tile_payload.get("row_padding"), int)
                else None,
                orientation=tile_payload.get("orientation")
                if isinstance(tile_payload.get("orientation"), int)
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


def _render_raster_background_data_uri(
    data: bytes,
) -> tuple[str, int, int] | None:
    from cgm.extract.raster import extract_raw_images_from_bytes  # noqa: PLC0415

    tiled = _render_first_tile_array(data)
    if tiled is not None:
        href = _image_to_png_data_uri(tiled)
        if href is not None:
            return href, tiled.width, tiled.height

    class29_raster = _render_first_element29_raster(data)
    if class29_raster is not None:
        href = _image_to_png_data_uri(class29_raster)
        if href is not None:
            return href, class29_raster.width, class29_raster.height

    indexed_palette = indexed_palette_bytes(extract_color_table(data))
    color_value_extent = extract_color_value_extent(data)

    for image in extract_raw_images_from_bytes(data):
        rendered = render_raw_image_payload(
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


def extract_vector_svg_from_bytes(
    data: bytes,
) -> str:
    """Convert CGM vector primitives and raster candidates into SVG.

    Raster precedence is: explicit tile-array overlays, id-29 multi-tile
    overlays, id-29 single-raster background, then Cell Array raster fallback.
    """
    style = _SvgStyle()
    color_table = extract_color_table(data)
    text_items: list[tuple[float, float, str, str, float]] = []
    restricted_text_items: list[tuple[float, float, float, float, str, str, float]] = []
    polyline_items: list[tuple[list[tuple[float, float]], str, float]] = []
    polygon_items: list[tuple[list[tuple[float, float]], str, float]] = []
    marker_items: list[tuple[float, float, str, float]] = []
    rectangle_items: list[tuple[float, float, float, float, str, float]] = []
    circle_items: list[tuple[float, float, float, str, float]] = []
    ellipse_items: list[tuple[float, float, float, float, str, float]] = []
    gdp_parameters: list[bytes] = []
    vdc_extent: tuple[float, float, float, float] | None = None
    descriptor_profile = extract_descriptor_profile(data)
    raster_tile_overlays = _render_first_tile_array_overlays(data)
    element29_overlays = (
        None if raster_tile_overlays is not None else _render_element29_overlays(data)
    )
    raster_background = (
        None
        if raster_tile_overlays is not None or element29_overlays is not None
        else _render_raster_background_data_uri(data)
    )
    transparency_enabled = True
    clip_enabled = False
    clip_rectangle: tuple[float, float, float, float] | None = None

    for element in iter_elements(data):
        if element.class_id == 2 and element.element_id == 6:
            decoded_vdc = decode_vdc_extent(element.parameters)
            if decoded_vdc is not None:
                vdc_extent = decoded_vdc
            continue

        if element.class_id == 3 and element.element_id == 4:
            mode = decode_transparency_mode(element.parameters)
            if isinstance(mode, bool):
                transparency_enabled = mode
            continue

        if element.class_id == 3 and element.element_id == 5:
            decoded_clip = decode_vdc_extent(element.parameters)
            if decoded_clip is not None:
                clip_rectangle = decoded_clip
            continue

        if element.class_id == 3 and element.element_id == 6:
            indicator = decode_clip_indicator(element.parameters)
            if isinstance(indicator, bool):
                clip_enabled = indicator
            continue

        if element.class_id == 5:
            if element.element_id == 3 and len(element.parameters) >= 4:
                width = parse_f32_be(element.parameters[:4])
                if width is not None and width > 0:
                    style.stroke_width = max(0.25, width)
            elif element.element_id == 4 and element.parameters:
                style.stroke_color = palette_index_to_hex(
                    element.parameters[0],
                    color_table=color_table,
                )
            elif element.element_id == 15 and len(element.parameters) >= 4:
                char_height = parse_f32_be(element.parameters[:4])
                if char_height is not None and char_height > 0:
                    style.font_size = max(6.0, char_height)
            continue

        if element.class_id != 4:
            continue

        if element.element_id == 1:
            points = decode_point_pairs_from_profile(
                element.parameters,
                profile=descriptor_profile,
            )
            if len(points) >= 2:
                polyline_items.append((points, style.stroke_color, style.stroke_width))
        elif element.element_id == 2:
            for start, end in decode_pairwise_segments(
                element.parameters,
                profile=descriptor_profile,
            ):
                polyline_items.append(([start, end], style.stroke_color, style.stroke_width))
        elif element.element_id == 3:
            points = decode_point_pairs_from_profile(
                element.parameters,
                profile=descriptor_profile,
            )
            for marker_x, marker_y in points:
                marker_items.append((marker_x, marker_y, style.stroke_color, style.marker_size))
        elif element.element_id == 4:
            appended = decode_cgm_text(element.parameters)
            if appended and text_items:
                last_x, last_y, last_text, last_color, last_font = text_items[-1]
                text_items[-1] = (last_x, last_y, f"{last_text}{appended}", last_color, last_font)
        elif element.element_id == 7:
            points = decode_point_pairs_from_profile(
                element.parameters,
                profile=descriptor_profile,
            )
            if len(points) >= 3:
                polygon_items.append((points, style.stroke_color, style.stroke_width))
        elif element.element_id == 8:
            points = decode_point_pairs_from_profile(
                element.parameters,
                profile=descriptor_profile,
            )
            if len(points) >= 3:
                polygon_items.append((points, style.stroke_color, style.stroke_width))
        elif element.element_id == 5:
            xy = decode_xy(
                element.parameters,
                profile=descriptor_profile,
            )
            if xy is None:
                continue
            x, y = xy
            text = decode_cgm_text(element.parameters[8:])
            if text is None:
                text = decode_cgm_text(element.parameters[4:])
            if text:
                text_items.append((x, y, text, style.stroke_color, style.font_size))
        elif element.element_id == 6:
            appended = decode_cgm_text(element.parameters)
            if appended and text_items:
                last_x, last_y, last_text, last_color, last_font = text_items[-1]
                text_items[-1] = (last_x, last_y, f"{last_text}{appended}", last_color, last_font)
        elif element.element_id in (10, 26):
            gdp_parameters.append(element.parameters)
        elif element.element_id == 11:
            corners = decode_rectangle_corners(
                element.parameters,
                profile=descriptor_profile,
            )
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
            circle = decode_circle_data(element.parameters, profile=descriptor_profile)
            if circle is None:
                continue
            cx, cy, radius = circle
            circle_items.append((cx, cy, radius, style.stroke_color, style.stroke_width))
        elif element.element_id in (13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24, 25, 27):
            arc_points = decode_point_pairs_from_profile(
                element.parameters,
                profile=descriptor_profile,
            )
            if len(arc_points) >= 2:
                polyline_items.append((arc_points, style.stroke_color, style.stroke_width))
        elif element.element_id == 28:
            continue
        elif element.element_id == 9:
            continue
        elif element.element_id == 17:
            ellipse = decode_ellipse_data(
                element.parameters,
                profile=descriptor_profile,
            )
            if ellipse is None:
                continue
            cx, cy, rx, ry = ellipse
            ellipse_items.append((cx, cy, rx, ry, style.stroke_color, style.stroke_width))
        elif element.element_id == 29:
            restricted = decode_restricted_text(element.parameters)
            if restricted is not None:
                anchor_x, anchor_y, box_w, box_h, text = restricted
                restricted_text_items.append(
                    (anchor_x, anchor_y, box_w, box_h, text, style.stroke_color, style.font_size)
                )

    for parameters in gdp_parameters:
        points = decode_gdp_polyline_points(
            parameters,
            profile=descriptor_profile,
        )
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
    elif element29_overlays is not None:
        _tile_entries, raster_w, raster_h = element29_overlays
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
    elif element29_overlays is not None:
        tile_entries, tile_total_w, tile_total_h = element29_overlays
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

    svg_lines.append("</svg>")
    return "\n".join(svg_lines) + "\n"


def extract_vector_svg(
    file_path: str | Path,
) -> str:
    """Convert a CGM file to SVG using the supported vector and raster paths."""
    path = Path(file_path)
    raw = path.read_bytes()
    if log.isEnabledFor(logging.DEBUG):
        log.debug("Loaded CGM file %s (%d bytes) for vector SVG conversion", path, len(raw))
    return extract_vector_svg_from_bytes(
        raw,
    )


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
    svg = extract_vector_svg(
        file_path,
    )
    target.write_text(svg, encoding="utf-8")
    if log.isEnabledFor(logging.DEBUG):
        log.debug("Wrote vector SVG: %s", target)
    return target


__all__ = [
    "extract_vector_svg",
    "extract_vector_svg_from_bytes",
    "extract_vector_svg_to_directory",
]
