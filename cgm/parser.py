"""Low-level parser for binary and clear-text CGM files (ISO/IEC 8632)."""

from __future__ import annotations

__author__ = "Kestin Goforth"
__copyright__ = "Copyright 2026"
__license__ = "BSD-3-Clause"

import logging
import re
import struct
from typing import TYPE_CHECKING

from .errors import CGMParseError
from .types import CGMElement

if TYPE_CHECKING:
    from collections.abc import Iterator


LONG_FORM_SENTINEL = 0x1F


# Class 4, Element 9 is Cell Array (raster data carrier in CGM).
CELL_ARRAY_CLASS_ID = 4
CELL_ARRAY_ELEMENT_ID = 9

log = logging.getLogger("cgm.parser")


def _read_u16_be(data: bytes, offset: int) -> int:
    if offset + 2 > len(data):
        raise CGMParseError("Unexpected end-of-file while reading 16-bit value")
    return int.from_bytes(data[offset : offset + 2], "big")


def _read_parameter_block(data: bytes, offset: int, declared_length: int) -> tuple[bytes, int]:
    """Read either short-form or long-form CGM element parameters."""
    if declared_length != LONG_FORM_SENTINEL:
        end = offset + declared_length
        if end > len(data):
            raise CGMParseError("Unexpected end-of-file in short-form element parameters")

        params = data[offset:end]
        next_offset = end + (declared_length & 1)
        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                "Short-form params: start=%d length=%d padded=%s next=%d",
                offset,
                declared_length,
                bool(declared_length & 1),
                next_offset,
            )
        return params, next_offset

    chunks: list[bytes] = []
    cursor = offset
    if log.isEnabledFor(logging.DEBUG):
        log.debug("Long-form params: start=%d", offset)

    while True:
        part_header = _read_u16_be(data, cursor)
        cursor += 2

        has_more = bool(part_header & 0x8000)
        part_length = part_header & 0x7FFF

        end = cursor + part_length
        if end > len(data):
            raise CGMParseError("Unexpected end-of-file in long-form element parameters")

        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                "Long-form chunk: cursor=%d length=%d has_more=%s",
                cursor,
                part_length,
                has_more,
            )

        chunks.append(data[cursor:end])
        cursor = end

        if part_length & 1:
            cursor += 1

        if not has_more:
            break

    params = b"".join(chunks)
    if log.isEnabledFor(logging.DEBUG):
        log.debug("Long-form params complete: total_length=%d next=%d", len(params), cursor)
    return params, cursor


def _iter_binary_elements(data: bytes) -> Iterator[CGMElement]:
    """Yield parsed elements from a binary CGM byte stream."""
    offset = 0
    size = len(data)

    if log.isEnabledFor(logging.DEBUG):
        log.debug("Begin binary element iteration: stream_size=%d", size)

    while offset < size:
        element_offset = offset
        header = _read_u16_be(data, offset)
        offset += 2

        class_id = (header >> 12) & 0x0F
        element_id = (header >> 5) & 0x7F
        declared_length = header & 0x1F

        parameters, offset = _read_parameter_block(data, offset, declared_length)

        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                (
                    "Element parsed: offset=%d class=%d id=%d declared_length=%d "
                    "actual_params=%d next_offset=%d"
                ),
                element_offset,
                class_id,
                element_id,
                declared_length,
                len(parameters),
                offset,
            )

        yield CGMElement(
            class_id=class_id,
            element_id=element_id,
            parameters=parameters,
            offset=element_offset,
        )


def _looks_like_text_cgm(data: bytes) -> bool:
    if not data:
        return False

    head = data[:512]
    if b"\x00" in head:
        return False

    # Treat streams with many control bytes as binary.
    control_bytes = sum(byte < 9 or (13 < byte < 32) for byte in head)
    if control_bytes > max(4, len(head) // 20):
        return False

    try:
        sample = head.decode("ascii")
    except UnicodeDecodeError:
        return False

    upper = sample.upper()
    if ";" not in upper:
        return False

    keywords = (
        "BEGMF",
        "LINE",
        "POLYGON",
        "TEXT",
        "CELLARRAY",
        "BEGTILEARRAY",
        "BITONALTILE",
        "APD",
        "BEGAPS",
        "ENDAPS",
    )
    return any(token in upper for token in keywords)


def _split_text_commands(text: str) -> Iterator[tuple[int, str]]:
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
            statement = text[start:idx].strip()
            if statement:
                yield start, statement
            start = idx + 1

    tail = text[start:].strip()
    if tail:
        yield start, tail


_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
_QUOTED_RE = re.compile(r"\"([^\"]*)\"|'([^']*)'")


def _extract_numbers(statement: str) -> list[float]:
    return [float(item) for item in _NUMBER_RE.findall(statement)]


def _extract_strings(statement: str) -> list[str]:
    strings: list[str] = []
    for match in _QUOTED_RE.finditer(statement):
        quoted = match.group(1) if match.group(1) is not None else match.group(2)
        if quoted is not None:
            strings.append(quoted)
    return strings


def _pack_f32_points(numbers: list[float]) -> bytes:
    values = numbers if len(numbers) % 2 == 0 else numbers[:-1]
    return b"".join(struct.pack(">f", value) for value in values)


def _pack_u16(value: int) -> bytes:
    return max(0, min(65535, value)).to_bytes(2, "big", signed=False)


def _build_apd_property(key: str, value: bytes) -> bytes:
    key_bytes = key.encode("ascii", errors="ignore")[:255]
    if len(value) <= 255:
        return bytes([len(key_bytes)]) + key_bytes + bytes([len(value)]) + value
    return bytes([len(key_bytes)]) + key_bytes + len(value).to_bytes(2, "big") + value


def _build_synthetic_cell_array_parameters(
    payload: bytes,
    width: int | None,
    height: int | None,
) -> bytes:
    """Build a Cell Array-like parameter block for extracted tile payload bytes."""
    nx = _pack_u16(width if width and width > 0 else 1)
    ny = _pack_u16(height if height and height > 0 else 1)
    local_color_precision = _pack_u16(1)
    cell_representation_mode = _pack_u16(0)
    return (b"\x00" * 12) + nx + ny + local_color_precision + cell_representation_mode + payload


_HEX_RUN_RE = re.compile(r"[0-9A-Fa-f]{32,}")


def _extract_hex_payload(statement: str) -> bytes:
    """Extract large hexadecimal runs from a text statement and decode to bytes."""
    runs = _HEX_RUN_RE.findall(statement)
    if not runs:
        return b""

    combined = "".join(runs)
    if len(combined) & 1:
        combined = combined[:-1]

    if not combined:
        return b""

    try:
        return bytes.fromhex(combined)
    except ValueError:
        return b""


def _derive_tile_dimensions(numbers: list[float]) -> tuple[int | None, int | None]:
    """Best-effort extraction of tile dimensions from BEGTILEARRAY parameters."""
    ints = [round(value) for value in numbers]
    if len(ints) >= 14 and ints[-2] > 0 and ints[-1] > 0:
        return ints[-2], ints[-1]
    if len(ints) >= 8 and ints[6] > 0 and ints[7] > 0:
        return ints[6], ints[7]
    return None, None


def _map_text_command(statement: str) -> tuple[int, int, bytes] | None:
    words = statement.split()
    if not words:
        return None

    command = words[0].upper()
    strings = _extract_strings(statement)
    numbers = _extract_numbers(statement)

    if command in {"LINE", "POLYLINE"}:
        return 4, 1, _pack_f32_points(numbers)

    if command == "POLYGON":
        return 4, 7, _pack_f32_points(numbers)

    if command in {"VDCEXT", "VDCEXTENT"} and len(numbers) >= 4:
        x1, y1, x2, y2 = (round(numbers[idx]) for idx in range(4))
        params = _pack_u16(x1) + _pack_u16(y1) + _pack_u16(x2) + _pack_u16(y2)
        return 2, 6, params

    if command == "TEXT" and len(numbers) >= 2 and strings:
        x = struct.pack(">f", numbers[0])
        y = struct.pack(">f", numbers[1])
        text_bytes = strings[-1].encode("ascii", errors="ignore")[:255]
        return 4, 5, x + y + bytes([len(text_bytes)]) + text_bytes

    if command in {"RESTRICTEDTEXT", "RESTRTEXT"} and len(numbers) >= 4 and strings:
        box_w = _pack_u16(round(numbers[0]))
        box_h = _pack_u16(round(numbers[1]))
        anchor_x = _pack_u16(round(numbers[2]))
        anchor_y = _pack_u16(round(numbers[3]))
        text_bytes = strings[-1].encode("ascii", errors="ignore")[:255]
        return 4, 29, box_w + box_h + anchor_x + anchor_y + bytes([len(text_bytes)]) + text_bytes

    if command == "CELLARRAY" and len(numbers) >= 9:
        x1, y1, x2, y2, x3, y3 = (round(numbers[idx]) for idx in range(6))
        nx = max(1, round(numbers[6]))
        ny = max(1, round(numbers[7]))
        lcp = round(numbers[8])
        pixels = bytes(max(0, min(255, round(value))) for value in numbers[9:])
        params = (
            _pack_u16(x1)
            + _pack_u16(y1)
            + _pack_u16(x2)
            + _pack_u16(y2)
            + _pack_u16(x3)
            + _pack_u16(y3)
            + _pack_u16(nx)
            + _pack_u16(ny)
            + _pack_u16(lcp)
            + _pack_u16(0)
            + pixels
        )
        return 4, 9, params

    if command in {"BEGAPS", "BEGINAPS", "BEGINAPPLICATIONSTRUCTURE"}:
        tag = (strings[0] if strings else "").encode("ascii", errors="ignore")[:255]
        return 0, 21, bytes([len(tag)]) + tag

    if command in {"ENDAPS", "ENDAPPLICATIONSTRUCTURE"}:
        return 0, 22, b""

    if command in {"APD", "APPLICATIONDATA"}:
        if len(strings) >= 2:
            key = strings[0]
            value_text = strings[1]
        else:
            # Fallback: APD key value
            bare = [token.strip(",()") for token in words[1:] if token.strip(",()")]
            if len(bare) < 2:
                return None
            key = bare[0]
            value_text = " ".join(bare[1:])

        key_norm = key.strip().lower()
        if key_norm == "region":
            region_numbers = _extract_numbers(value_text)
            if len(region_numbers) >= 4:
                x_min, y_min, x_max, y_max = (round(region_numbers[idx]) for idx in range(4))
                value = _pack_u16(x_min) + _pack_u16(y_min) + _pack_u16(x_max) + _pack_u16(y_max)
            else:
                value = value_text.encode("ascii", errors="ignore")
        else:
            value = value_text.encode("ascii", errors="ignore")

        return 9, 1, _build_apd_property(key_norm, value)

    return None


def _iter_text_elements(data: bytes) -> Iterator[CGMElement]:
    """Yield parsed elements from clear-text CGM command streams."""
    text = data.decode("latin-1")
    tile_dims: tuple[int | None, int | None] = (None, None)

    if log.isEnabledFor(logging.DEBUG):
        log.debug("Begin clear-text element iteration: stream_size=%d", len(data))

    for command_offset, statement in _split_text_commands(text):
        words = statement.split()
        if not words:
            continue

        command = words[0].upper()

        if command == "BEGTILEARRAY":
            tile_dims = _derive_tile_dimensions(_extract_numbers(statement))
            continue

        if command == "ENDTILEARRAY":
            tile_dims = (None, None)
            continue

        if command == "BITONALTILE":
            payload = _extract_hex_payload(statement)
            if not payload:
                continue
            width, height = tile_dims
            parameters = _build_synthetic_cell_array_parameters(payload, width, height)
            yield CGMElement(
                class_id=CELL_ARRAY_CLASS_ID,
                element_id=CELL_ARRAY_ELEMENT_ID,
                parameters=parameters,
                offset=command_offset,
            )
            continue

        mapped = _map_text_command(statement)
        if mapped is None:
            continue
        class_id, element_id, parameters = mapped
        yield CGMElement(
            class_id=class_id,
            element_id=element_id,
            parameters=parameters,
            offset=command_offset,
        )


def iter_elements(data: bytes) -> Iterator[CGMElement]:
    """Yield parsed elements from either binary or clear-text CGM streams."""
    if _looks_like_text_cgm(data):
        yield from _iter_text_elements(data)
        return

    yield from _iter_binary_elements(data)
