"""Shared CGM decoding helpers used by extraction modules."""

from __future__ import annotations

import logging
import math
import struct

log = logging.getLogger("cgm.common")


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


def _decode_prefixed_ascii(parameters: bytes) -> str | None:
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
    if len(value) < 8:
        return None
    x_min = int.from_bytes(value[-8:-6], "big", signed=False)
    y_min = int.from_bytes(value[-6:-4], "big", signed=False)
    x_max = int.from_bytes(value[-4:-2], "big", signed=False)
    y_max = int.from_bytes(value[-2:], "big", signed=False)
    if x_min >= x_max or y_min >= y_max:
        return None
    return x_min, y_min, x_max, y_max


def _parse_cell_array_hints(parameters: bytes) -> tuple[int | None, int | None, int]:
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


def _decode_restricted_text(parameters: bytes) -> tuple[float, float, float, float, str] | None:
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

        if abs(box_w) > 1_000_000 or abs(box_h) > 1_000_000:
            continue

        return float(anchor_x), float(anchor_y), float(box_w), float(box_h), text

    return None
