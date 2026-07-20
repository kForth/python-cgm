"""High-level CGM extraction helpers for SVG, tile/raster decoding, and hotspot recovery."""

from __future__ import annotations

__author__ = "Kestin Goforth"
__copyright__ = "Copyright 2026"
__license__ = "BSD-3-Clause"

import logging
import math
import struct
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING

from PIL import Image

from cgm.parser import iter_elements

if TYPE_CHECKING:
    from cgm.types import RawImage

log = logging.getLogger("cgm.extract")


def candidate_signature(candidate: dict[str, object] | None) -> tuple[str, str, bool] | None:
    """Return normalized signature tuple for a decode candidate entry."""
    if not isinstance(candidate, dict):
        return None
    decoder = candidate.get("decoder")
    variant = candidate.get("encoded_variant")
    invert = candidate.get("invert")
    if not isinstance(decoder, str) or not isinstance(variant, str) or not isinstance(invert, bool):
        return None
    return decoder, variant, invert


def choose_consensus_decode_signature(
    decoded_tile_meta: dict[int, dict[str, object]],
) -> tuple[str, str, bool] | None:
    """Choose dominant decode signature across all successfully decoded tiles."""
    signatures: list[tuple[str, str, bool]] = []
    for meta in decoded_tile_meta.values():
        candidate = meta.get("best_candidate") if isinstance(meta, dict) else None
        signature = candidate_signature(candidate if isinstance(candidate, dict) else None)
        if signature is not None:
            signatures.append(signature)

    if not signatures:
        return None

    counts = Counter(signatures)
    return counts.most_common(1)[0][0]


def choose_consensus_decode_dimensions(
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


def parse_cell_array_hints(parameters: bytes) -> tuple[int | None, int | None, int]:
    """Parse common Cell Array metadata for binary CGM streams when the layout matches.

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


def parse_cell_array_metadata(parameters: bytes) -> dict[str, int | None]:
    """Parse common binary Cell Array metadata in a safe, profile-agnostic way."""
    width, height, payload_offset = parse_cell_array_hints(parameters)
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
class _CgmDescriptorProfile:
    vdc_type: int | None = None
    vdc_integer_precision: int | None = None
    vdc_real_precision_bits: int | None = None


def extract_descriptor_profile(data: bytes) -> _CgmDescriptorProfile:
    """Extract a minimal CGM descriptor profile used for strict coordinate decode."""

    profile = _CgmDescriptorProfile()
    for element in iter_elements(data):
        if element.class_id != 1:
            continue

        if element.element_id == 3:
            # VDC type: INTEGER=0, REAL=1 in common profiles.
            if len(element.parameters) >= 2:
                profile.vdc_type = int.from_bytes(element.parameters[:2], "big", signed=False)
            elif element.parameters:
                profile.vdc_type = int(element.parameters[0])
            continue

        if element.element_id == 11:
            # VDC integer precision in bits.
            if len(element.parameters) >= 2:
                profile.vdc_integer_precision = int.from_bytes(
                    element.parameters[:2],
                    "big",
                    signed=False,
                )
            continue

        if element.element_id == 12:
            # VDC real precision descriptors vary by profile; keep a compact
            # hint for common IEEE-like 32/64-bit paths.
            if len(element.parameters) >= 2:
                head = int.from_bytes(element.parameters[:2], "big", signed=False)
                if head in (32, 64):
                    profile.vdc_real_precision_bits = head
                elif len(element.parameters) >= 6:
                    mantissa_bits = int.from_bytes(element.parameters[4:6], "big", signed=False)
                    if mantissa_bits >= 52:
                        profile.vdc_real_precision_bits = 64
                    elif mantissa_bits >= 23:
                        profile.vdc_real_precision_bits = 32
            continue

    return profile


def parse_f32_be(value: bytes) -> float | None:
    if len(value) != 4:
        return None
    parsed = float(struct.unpack(">f", value)[0])
    if not math.isfinite(parsed):
        return None
    return parsed


def parse_f64_be(value: bytes) -> float | None:
    if len(value) != 8:
        return None
    parsed = float(struct.unpack(">d", value)[0])
    if not math.isfinite(parsed):
        return None
    return parsed


def decode_point_pairs_exact(parameters: bytes) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for offset in range(0, len(parameters) - 7, 8):
        x = parse_f32_be(parameters[offset : offset + 4])
        y = parse_f32_be(parameters[offset + 4 : offset + 8])
        if x is None or y is None:
            continue
        points.append((x, y))
    return points


def decode_point_pairs_i32(parameters: bytes) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for offset in range(0, len(parameters) - 7, 8):
        x = int.from_bytes(parameters[offset : offset + 4], "big", signed=True)
        y = int.from_bytes(parameters[offset + 4 : offset + 8], "big", signed=True)
        points.append((float(x), float(y)))
    return points


def decode_cgm_text(parameters: bytes) -> str | None:
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


def decode_prefixed_ascii(parameters: bytes) -> str | None:
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


def decode_application_property(parameters: bytes) -> tuple[str, bytes] | None:
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


def decode_hotspot_region_bbox(value: bytes) -> tuple[int, int, int, int] | None:
    """Extract a rectangular region from hotspot payload bytes when one is present."""
    if len(value) < 8:
        return None
    x_min = int.from_bytes(value[-8:-6], "big", signed=False)
    y_min = int.from_bytes(value[-6:-4], "big", signed=False)
    x_max = int.from_bytes(value[-4:-2], "big", signed=False)
    y_max = int.from_bytes(value[-2:], "big", signed=False)
    if x_min >= x_max or y_min >= y_max:
        return None
    return x_min, y_min, x_max, y_max


def default_palette_entry(color_index: int) -> tuple[int, int, int]:
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


def palette_index_to_hex(
    color_index: int,
    *,
    color_table: dict[int, tuple[int, int, int]] | None = None,
) -> str:
    rgb = (color_table or {}).get(color_index, default_palette_entry(color_index))
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def decode_transparency_mode(parameters: bytes) -> bool | None:
    """Return transparency mode (True=on, False=off) when decodable."""
    if not parameters:
        return None
    if len(parameters) >= 2:
        value = int.from_bytes(parameters[:2], "big", signed=False)
    else:
        value = parameters[0]
    return value != 0


def decode_clip_indicator(parameters: bytes) -> bool | None:
    """Return clip indicator (True=on, False=off) when decodable."""
    if not parameters:
        return None
    if len(parameters) >= 2:
        value = int.from_bytes(parameters[:2], "big", signed=False)
    else:
        value = parameters[0]
    return value != 0


def decode_color_table_parameters(parameters: bytes) -> dict[int, tuple[int, int, int]]:
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
                    r = scale_component_to_byte(r16, bits=16)
                    g = scale_component_to_byte(g16, bits=16)
                    b = scale_component_to_byte(b16, bits=16)

                table[start_index + entry_idx] = (r, g, b)
                cursor += entry_size

    return table


def scale_component_to_byte(value: int, *, bits: int) -> int:
    if bits <= 8:
        return max(0, min(255, value))
    max_value = (1 << bits) - 1
    if max_value <= 0:
        return 0
    return max(0, min(255, round((value / max_value) * 255.0)))


def extract_color_table(data: bytes) -> dict[int, tuple[int, int, int]]:
    table: dict[int, tuple[int, int, int]] = {}
    for element in iter_elements(data):
        if element.class_id == 5 and element.element_id == 34:
            table.update(decode_color_table_parameters(element.parameters))
    return table


def decode_color_value_extent_parameters(
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


def extract_color_value_extent(data: bytes) -> tuple[int, int, int, int, int, int] | None:
    extent: tuple[int, int, int, int, int, int] | None = None
    for element in iter_elements(data):
        if element.class_id == 1 and element.element_id == 10:
            decoded = decode_color_value_extent_parameters(element.parameters)
            if decoded is not None:
                extent = decoded
    return extent


def scale_direct16_rgb_payload(
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


def indexed_palette_bytes(color_table: dict[int, tuple[int, int, int]]) -> bytes:
    channels: list[int] = []
    for index in range(256):
        r, g, b = color_table.get(index, default_palette_entry(index))
        channels.extend((r, g, b))
    return bytes(channels)


def decode_i16_be(value: bytes) -> int | None:
    if len(value) != 2:
        return None
    return int.from_bytes(value, "big", signed=True)


def decode_i32_be(value: bytes) -> int | None:
    if len(value) != 4:
        return None
    return int.from_bytes(value, "big", signed=True)


def decode_point_pairs_i16(parameters: bytes) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for offset in range(0, len(parameters) - 3, 4):
        x = decode_i16_be(parameters[offset : offset + 2])
        y = decode_i16_be(parameters[offset + 2 : offset + 4])
        if x is None or y is None:
            continue
        points.append((float(x), float(y)))
    return points


def decode_point_pairs_f64(parameters: bytes) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for offset in range(0, len(parameters) - 15, 16):
        x = parse_f64_be(parameters[offset : offset + 8])
        y = parse_f64_be(parameters[offset + 8 : offset + 16])
        if x is None or y is None:
            continue
        points.append((x, y))
    return points


def decode_point_pairs_from_profile(
    parameters: bytes,
    *,
    profile: _CgmDescriptorProfile,
) -> list[tuple[float, float]]:
    """Decode points using declared CGM descriptor values when available."""

    if profile.vdc_type == 0:
        precision = profile.vdc_integer_precision
        if precision is not None and precision <= 16:
            return decode_point_pairs_i16(parameters)
        if precision is not None and precision > 16:
            return decode_point_pairs_i32(parameters)
        # Integer VDC without precision descriptor defaults to a common 16-bit layout.
        return decode_point_pairs_i16(parameters)

    if profile.vdc_type == 1:
        if profile.vdc_real_precision_bits == 64:
            return decode_point_pairs_f64(parameters)
        return decode_point_pairs_exact(parameters)

    # Unknown VDC descriptor profile: do not guess mixed encodings.
    return []


def decode_vdc_extent(parameters: bytes) -> tuple[float, float, float, float] | None:
    if len(parameters) >= 16:
        real_values: list[float] = []
        for idx in range(0, 16, 4):
            value = parse_f32_be(parameters[idx : idx + 4])
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


def decode_xy(
    parameters: bytes,
    *,
    profile: _CgmDescriptorProfile,
) -> tuple[float, float] | None:
    if profile.vdc_type == 0:
        if profile.vdc_integer_precision is not None and profile.vdc_integer_precision <= 16:
            if len(parameters) >= 4:
                x = decode_i16_be(parameters[:2])
                y = decode_i16_be(parameters[2:4])
                if x is not None and y is not None:
                    return float(x), float(y)
            return None
        if len(parameters) >= 8:
            x = decode_i32_be(parameters[:4])
            y = decode_i32_be(parameters[4:8])
            if x is not None and y is not None:
                return float(x), float(y)
        return None

    if profile.vdc_type == 1:
        if profile.vdc_real_precision_bits == 64:
            if len(parameters) >= 16:
                x_f64 = parse_f64_be(parameters[:8])
                y_f64 = parse_f64_be(parameters[8:16])
                if x_f64 is not None and y_f64 is not None:
                    return x_f64, y_f64
            return None
        if len(parameters) >= 8:
            x_f32 = parse_f32_be(parameters[:4])
            y_f32 = parse_f32_be(parameters[4:8])
            if x_f32 is not None and y_f32 is not None:
                return x_f32, y_f32
        return None

    return None


def decode_pairwise_segments(
    parameters: bytes,
    *,
    profile: _CgmDescriptorProfile,
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    points = decode_point_pairs_from_profile(parameters, profile=profile)
    segments: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for idx in range(0, len(points) - 1, 2):
        segments.append((points[idx], points[idx + 1]))
    return segments


def decode_gdp_polyline_points(
    parameters: bytes,
    *,
    profile: _CgmDescriptorProfile,
) -> list[tuple[float, float]]:
    """Decode GDP payload coordinates using declared VDC descriptors only."""
    return decode_point_pairs_from_profile(parameters, profile=profile)


def decode_rectangle_corners(
    parameters: bytes,
    *,
    profile: _CgmDescriptorProfile,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    points = decode_point_pairs_from_profile(parameters, profile=profile)
    if len(points) >= 2:
        return points[0], points[1]
    return None


def decode_circle_data(
    parameters: bytes,
    *,
    profile: _CgmDescriptorProfile,
) -> tuple[float, float, float] | None:
    if profile.vdc_type == 0:
        if profile.vdc_integer_precision is not None and profile.vdc_integer_precision <= 16:
            if len(parameters) < 6:
                return None
            cx_i16 = decode_i16_be(parameters[0:2])
            cy_i16 = decode_i16_be(parameters[2:4])
            radius_i16 = decode_i16_be(parameters[4:6])
            if cx_i16 is None or cy_i16 is None or radius_i16 is None or radius_i16 <= 0:
                return None
            return float(cx_i16), float(cy_i16), float(radius_i16)

        if len(parameters) < 12:
            return None
        cx_i32 = decode_i32_be(parameters[0:4])
        cy_i32 = decode_i32_be(parameters[4:8])
        radius_i32 = decode_i32_be(parameters[8:12])
        if cx_i32 is None or cy_i32 is None or radius_i32 is None or radius_i32 <= 0:
            return None
        return float(cx_i32), float(cy_i32), float(radius_i32)

    if profile.vdc_type == 1:
        if profile.vdc_real_precision_bits == 64:
            if len(parameters) < 24:
                return None
            cx_f64 = parse_f64_be(parameters[0:8])
            cy_f64 = parse_f64_be(parameters[8:16])
            radius_f64 = parse_f64_be(parameters[16:24])
            if cx_f64 is None or cy_f64 is None or radius_f64 is None or radius_f64 <= 0:
                return None
            return cx_f64, cy_f64, abs(radius_f64)

        if len(parameters) < 12:
            return None
        cx_f32 = parse_f32_be(parameters[0:4])
        cy_f32 = parse_f32_be(parameters[4:8])
        radius_f32 = parse_f32_be(parameters[8:12])
        if cx_f32 is None or cy_f32 is None or radius_f32 is None or radius_f32 <= 0:
            return None
        return cx_f32, cy_f32, abs(radius_f32)

    return None


def decode_ellipse_data(
    parameters: bytes,
    *,
    profile: _CgmDescriptorProfile,
) -> tuple[float, float, float, float] | None:
    points = decode_point_pairs_from_profile(parameters, profile=profile)
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


def decode_restricted_text(parameters: bytes) -> tuple[float, float, float, float, str] | None:
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
        text = decode_cgm_text(text_bytes)
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


def analyze_element29_payload(parameters: bytes) -> dict[str, object]:
    """Provide structured diagnostics for class-4/id-29 payloads."""
    byte_counts = Counter(parameters)
    top_bytes = [
        {"byte": f"0x{value:02x}", "count": count} for value, count in byte_counts.most_common(8)
    ]
    head_u16 = [
        int.from_bytes(parameters[idx : idx + 2], "big")
        for idx in range(0, min(64, len(parameters) - 1), 2)
    ]
    restricted = decode_restricted_text(parameters)
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


def render_raw_image_payload(
    image: RawImage,
    *,
    indexed_palette: bytes | None = None,
    color_value_extent: tuple[int, int, int, int, int, int] | None = None,
) -> Image.Image | None:
    if image.width is None or image.height is None:
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
        scaled_rgb = scale_direct16_rgb_payload(
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


def coerce_int(value: object, default: int = 1) -> int:
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
