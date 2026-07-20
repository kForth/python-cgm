"""Clear-text CGM command parsing and mapping."""

from __future__ import annotations

import logging
import re
import struct
from typing import TYPE_CHECKING

from cgm.parser.constants import _TEXT_TILE_COMMANDS, CELL_ARRAY_CLASS_ID, CELL_ARRAY_ELEMENT_ID
from cgm.types import CGMElement

if TYPE_CHECKING:
    from collections.abc import Iterator

log = logging.getLogger("cgm.parser")

_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
_QUOTED_RE = re.compile(r'"([^\"]*)"|\'([^\']*)\'')
_HEX_RUN_RE = re.compile(r"[0-9A-Fa-f]{32,}")


def split_text_commands(text: str) -> Iterator[tuple[int, str]]:
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


def _pack_f32_values(numbers: list[float]) -> bytes:
    return b"".join(struct.pack(">f", value) for value in numbers)


def _pack_u16(value: int) -> bytes:
    return max(0, min(65535, value)).to_bytes(2, "big", signed=False)


def _color_component_to_byte(value: float) -> int:
    """Normalize color component to 0..255 for synthetic color-table records."""
    if 0.0 <= value <= 1.0:
        return max(0, min(255, round(value * 255.0)))
    return max(0, min(255, round(value)))


def _build_apd_property(key: str, value: bytes) -> bytes:
    key_bytes = key.encode("ascii", errors="ignore")[:255]
    if len(value) <= 255:
        return bytes([len(key_bytes)]) + key_bytes + bytes([len(value)]) + value
    return bytes([len(key_bytes)]) + key_bytes + len(value).to_bytes(2, "big") + value


def _build_synthetic_cell_array_parameters(
    payload: bytes,
    width: int | None,
    height: int | None,
    *,
    local_color_precision: int | None = None,
    cell_representation_mode: int | None = None,
) -> bytes:
    """Build a Cell Array-like parameter block for extracted tile payload bytes."""
    nx = _pack_u16(width if width and width > 0 else 1)
    ny = _pack_u16(height if height and height > 0 else 1)
    lcp = _pack_u16(local_color_precision if local_color_precision is not None else 1)
    mode = _pack_u16(cell_representation_mode if cell_representation_mode is not None else 0)
    return (b"\x00" * 12) + nx + ny + lcp + mode + payload


def _default_tile_local_color_precision(command: str) -> int:
    if command in {"BITONALTILE", "MONOCHROMETILE"}:
        return 1
    if command in {"INDEXCOLORTILE", "INDEXCOLOURTILE"}:
        return 8
    if command in {"COLORTILE", "COLOURTILE", "DIRECTCOLORTILE", "DIRECTCOLOURTILE"}:
        return 24
    return 1


def _extract_hex_payload(statement: str) -> bytes:
    """Extract large hexadecimal runs from a text statement and decode to bytes."""
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
    """Extract tile dimensions from BEGTILEARRAY parameters when they are present."""
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

    if command in {"DISJOINTPOLYLINE", "DISJTLINE", "DJPOLYLINE"}:
        return 4, 2, _pack_f32_points(numbers)

    if command in {"POLYMARKER", "MARKER"}:
        return 4, 3, _pack_f32_points(numbers)

    if command == "POLYGON":
        return 4, 7, _pack_f32_points(numbers)

    if command == "POLYGONSET":
        return 4, 8, _pack_f32_points(numbers)

    if command in {"VDCEXT", "VDCEXTENT"} and len(numbers) >= 4:
        x1, y1, x2, y2 = (round(numbers[idx]) for idx in range(4))
        params = _pack_u16(x1) + _pack_u16(y1) + _pack_u16(x2) + _pack_u16(y2)
        return 2, 6, params

    if command in {"COLRVALUEEXT", "COLORVALUEEXTENT", "COLOURVALUEEXTENT"} and len(numbers) >= 6:
        values = [max(0, min(65535, round(numbers[idx]))) for idx in range(6)]
        params = b"".join(_pack_u16(value) for value in values)
        return 1, 10, params

    if command in {"CLIPRECT", "CLIPRECTANGLE"} and len(numbers) >= 4:
        x1, y1, x2, y2 = (round(numbers[idx]) for idx in range(4))
        params = _pack_u16(x1) + _pack_u16(y1) + _pack_u16(x2) + _pack_u16(y2)
        return 3, 5, params

    if command in {"CLIPIND", "CLIPINDICATOR", "CLIP"}:
        enabled: int | None = None
        if len(words) >= 2:
            token = words[1].strip().upper()
            if token in {"ON", "TRUE"}:
                enabled = 1
            elif token in {"OFF", "FALSE"}:
                enabled = 0
        if enabled is None and numbers:
            enabled = 0 if round(numbers[0]) == 0 else 1
        if enabled is not None:
            return 3, 6, bytes([enabled])

    if command == "TEXT" and len(numbers) >= 2 and strings:
        x = struct.pack(">f", numbers[0])
        y = struct.pack(">f", numbers[1])
        text_bytes = strings[-1].encode("ascii", errors="ignore")[:255]
        return 4, 5, x + y + bytes([len(text_bytes)]) + text_bytes

    if command == "APPENDTEXT" and strings:
        text_bytes = strings[-1].encode("ascii", errors="ignore")[:255]
        return 4, 6, bytes([len(text_bytes)]) + text_bytes

    if command in {"RESTRICTEDTEXT", "RESTRTEXT"} and len(numbers) >= 4 and strings:
        box_w = _pack_u16(round(numbers[0]))
        box_h = _pack_u16(round(numbers[1]))
        anchor_x = _pack_u16(round(numbers[2]))
        anchor_y = _pack_u16(round(numbers[3]))
        text_bytes = strings[-1].encode("ascii", errors="ignore")[:255]
        return 4, 29, box_w + box_h + anchor_x + anchor_y + bytes([len(text_bytes)]) + text_bytes

    if command in {"RECT", "RECTANGLE"} and len(numbers) >= 4:
        return 4, 11, _pack_f32_values(numbers[:4])

    if command == "CIRCLE" and len(numbers) >= 3:
        return 4, 12, _pack_f32_values(numbers[:3])

    if command in {"ARC3PT", "CIRCULARARC3POINT", "CIRCULARARC3PT"} and len(numbers) >= 6:
        return 4, 13, _pack_f32_values(numbers[:6])

    if (
        command in {"ARC3PTCLOSE", "CIRCULARARC3POINTCLOSE", "CIRCULARARC3PTCLOSE"}
        and len(numbers) >= 6
    ):
        return 4, 14, _pack_f32_values(numbers[:6])

    if command in {"ARCCENTRE", "ARCCTR", "CIRCULARARCCENTRE"} and len(numbers) >= 6:
        return 4, 15, _pack_f32_values(numbers[:6])

    if command in {"ARCCENTRECLOSE", "ARCCTRCLOSE", "CIRCULARARCCENTRECLOSE"} and len(numbers) >= 6:
        return 4, 16, _pack_f32_values(numbers[:6])

    if command == "ELLIPSE" and len(numbers) >= 6:
        return 4, 17, _pack_f32_values(numbers[:6])

    if command in {"ELLIPTICALARC", "ELLIPARC"} and len(numbers) >= 10:
        return 4, 18, _pack_f32_values(numbers[:10])

    if command in {"ELLIPTICALARCCLOSE", "ELLIPARCCLOSE"} and len(numbers) >= 10:
        return 4, 19, _pack_f32_values(numbers[:10])

    if command in {"GDP", "GENERALISEDDRAWINGPRIMITIVE"}:
        return 4, 10, _pack_f32_values(numbers)

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

    if command in {"LINECOLR", "LINECOLOR", "LINECOLOUR"} and numbers:
        return 5, 4, bytes([max(0, min(255, round(numbers[0])))])

    if command in {"TRANSPARENCY", "TRANSPARENCYMODE", "TRANSPMODE"}:
        mode: int | None = None
        if len(words) >= 2:
            token = words[1].strip().upper()
            if token in {"ON", "TRUE"}:
                mode = 1
            elif token in {"OFF", "FALSE"}:
                mode = 0
        if mode is None and numbers:
            mode = 0 if round(numbers[0]) == 0 else 1
        if mode is not None:
            return 3, 4, bytes([mode])

    if command in {"COLRTABLE", "COLORTABLE", "COLOURTABLE"} and len(numbers) >= 4:
        start_index = max(0, min(255, round(numbers[0])))
        channels = numbers[1:]
        triplet_count = len(channels) // 3
        if triplet_count <= 0:
            return None
        payload = bytearray([start_index])
        for idx in range(triplet_count):
            base = idx * 3
            payload.extend(
                [
                    _color_component_to_byte(channels[base]),
                    _color_component_to_byte(channels[base + 1]),
                    _color_component_to_byte(channels[base + 2]),
                ]
            )
        return 5, 34, bytes(payload)

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


def iter_text_elements(data: bytes) -> Iterator[CGMElement]:
    """Yield parsed elements from clear-text CGM command streams."""
    text = data.decode("latin-1")
    text = re.sub(r"(^|;)\s*[\"'](?=[A-Za-z])", r"\1 ", text)
    tile_dims: tuple[int | None, int | None] = (None, None)

    if log.isEnabledFor(logging.DEBUG):
        log.debug("Begin clear-text element iteration: stream_size=%d", len(data))

    for command_offset, statement in split_text_commands(text):
        normalized_statement = statement.lstrip()
        while normalized_statement.startswith(('"', "'")):
            normalized_statement = normalized_statement[1:].lstrip()

        words = normalized_statement.split()
        if not words:
            continue

        command = words[0].upper()

        if command == "BEGTILEARRAY":
            tile_dims = _derive_tile_dimensions(_extract_numbers(normalized_statement))
            continue

        if command == "ENDTILEARRAY":
            tile_dims = (None, None)
            continue

        if command in _TEXT_TILE_COMMANDS:
            payload = _extract_hex_payload(normalized_statement)
            if not payload:
                continue
            width, height = tile_dims
            metadata_prefix = normalized_statement.split("''", 1)[0].split('""', 1)[0]
            numbers = _extract_numbers(metadata_prefix)
            local_color_precision = (
                round(numbers[1])
                if len(numbers) >= 2
                else _default_tile_local_color_precision(command)
            )
            cell_representation_mode = round(numbers[4]) if len(numbers) >= 5 else 0
            parameters = _build_synthetic_cell_array_parameters(
                payload,
                width,
                height,
                local_color_precision=local_color_precision,
                cell_representation_mode=cell_representation_mode,
            )
            yield CGMElement(
                class_id=CELL_ARRAY_CLASS_ID,
                element_id=CELL_ARRAY_ELEMENT_ID,
                parameters=parameters,
                offset=command_offset,
            )
            continue

        mapped = _map_text_command(normalized_statement)
        if mapped is None:
            continue
        class_id, element_id, parameters = mapped
        yield CGMElement(
            class_id=class_id,
            element_id=element_id,
            parameters=parameters,
            offset=command_offset,
        )
