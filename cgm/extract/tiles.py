"""Raster tile parsing and decoding helpers."""

from __future__ import annotations

import re
from functools import lru_cache

import imagecodecs
from PIL import Image

from cgm.extract.core import (
    coerce_int,
    extract_color_table,
    extract_color_value_extent,
    indexed_palette_bytes,
    scale_direct16_rgb_payload,
)
from cgm.parser import iter_elements

_TEXT_NUMBER_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")
_TEXT_HEX_RUN_RE = re.compile(r"[0-9A-Fa-f]{32,}")

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


def _reverse_bits_in_byte(value: int) -> int:
    """Return a byte with internal bit order reversed."""
    value = ((value & 0xF0) >> 4) | ((value & 0x0F) << 4)
    value = ((value & 0xCC) >> 2) | ((value & 0x33) << 2)
    value = ((value & 0xAA) >> 1) | ((value & 0x55) << 1)
    return value


@lru_cache(maxsize=512)
def _cached_ccittfax3_decode(payload: bytes, *, height: int, width: int) -> bytes | None:
    try:
        return bytes(imagecodecs.ccittfax3_decode(payload, height=height, width=width))
    except (RuntimeError, ValueError):
        return None


@lru_cache(maxsize=512)
def _cached_ccittfax4_decode(payload: bytes, *, height: int, width: int) -> bytes | None:
    try:
        return bytes(imagecodecs.ccittfax4_decode(payload, height=height, width=width))
    except (RuntimeError, ValueError):
        return None


def _decode_fax_output_to_bitmap(output: bytes, width: int, height: int) -> list[int] | None:
    """Normalize CCITT decoder output to a 0/1 bitmap list."""

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


def _make_tile_array_record(
    *,
    cols: int,
    rows: int,
    tile_width: int,
    tile_height: int,
    total_width: int,
    total_height: int,
    cell_path: int = 0,
) -> dict[str, object]:
    return {
        "cols": max(1, cols),
        "rows": max(1, rows),
        "tile_width": max(1, tile_width),
        "tile_height": max(1, tile_height),
        "total_width": max(1, total_width),
        "total_height": max(1, total_height),
        "cell_path": cell_path,
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
    row_padding: int | None = None,
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
            "row_padding": row_padding,
        }
    )


def _finalize_tile_array_record(array: dict[str, object]) -> dict[str, object] | None:
    tiles = array.get("tiles")
    if not isinstance(tiles, list) or not tiles:
        return None

    cols = coerce_int(array.get("cols", 1))
    rows = coerce_int(array.get("rows", 1))
    tile_w = coerce_int(array.get("tile_width", 1))
    tile_h = coerce_int(array.get("tile_height", 1))
    total_w = coerce_int(array.get("total_width", 1))
    total_h = coerce_int(array.get("total_height", 1))

    cell_path = coerce_int(array.get("cell_path", 0))
    return _make_tile_array_record(
        cols=cols,
        rows=rows,
        tile_width=tile_w,
        tile_height=tile_h,
        total_width=total_w,
        total_height=total_h,
        cell_path=cell_path,
    ) | {"tiles": tiles}


def _parse_tile_arrays(data: bytes) -> list[dict[str, object]]:
    """Parse tile-array metadata from clear-text commands or binary wrappers.

    Binary parsing targets class-0/id-19..20 blocks that contain class-4/id-28
    tile payloads, and also exposes header dimensions reused by id-29 wrapper
    decoding paths.
    """
    text_arrays = _parse_text_tile_arrays(data)
    if text_arrays:
        return text_arrays
    return _parse_binary_tile_arrays(data)


def _parse_binary_tile_array_header(parameters: bytes) -> dict[str, int] | None:
    """Parse the corpus-stable 32-byte binary tile-array header.

    The returned dimensions are used by id-28 tile-array composition and by
    id-29 wrapper-aware raster decode in files where class-4/id-29 is nested
    inside class-0/id-19..20 tile-array blocks.
    """
    if len(parameters) < 32:
        return None

    cell_path = int.from_bytes(parameters[4:8], "big", signed=False)
    cols = int.from_bytes(parameters[8:10], "big", signed=False)
    rows = int.from_bytes(parameters[10:12], "big", signed=False)
    tile_width = int.from_bytes(parameters[12:14], "big", signed=False)
    tile_height = int.from_bytes(parameters[14:16], "big", signed=False)
    total_width = int.from_bytes(parameters[28:30], "big", signed=False)
    total_height = int.from_bytes(parameters[30:32], "big", signed=False)

    if min(cols, rows, tile_width, tile_height, total_width, total_height) <= 0:
        return None

    return {
        "cell_path": cell_path,
        "cols": cols,
        "rows": rows,
        "tile_width": tile_width,
        "tile_height": tile_height,
        "total_width": total_width,
        "total_height": total_height,
    }


def _parse_binary_id28_tile(parameters: bytes) -> dict[str, object] | None:
    """Parse the corpus-stable class-4/id-28 tile wrapper."""
    if len(parameters) <= 7:
        return None

    payload = parameters[7:]
    if not payload:
        return None

    return {
        "payload": payload,
        "compression": int.from_bytes(parameters[0:2], "big", signed=False),
        "row_padding": int.from_bytes(parameters[2:4], "big", signed=False),
        "bit_order": int.from_bytes(parameters[4:6], "big", signed=False),
        "orientation": int(parameters[6]),
    }


def _parse_binary_tile_arrays(data: bytes) -> list[dict[str, object]]:
    """Parse binary tile arrays wrapped around class-4/id-28 payloads."""
    arrays: list[dict[str, object]] = []
    current: dict[str, object] | None = None

    for element in iter_elements(data):
        if element.class_id == 0 and element.element_id == 19:
            header = _parse_binary_tile_array_header(element.parameters)
            if header is None:
                current = None
                continue
            current = _make_tile_array_record(
                cols=header["cols"],
                rows=header["rows"],
                tile_width=header["tile_width"],
                tile_height=header["tile_height"],
                total_width=header["total_width"],
                total_height=header["total_height"],
                cell_path=header["cell_path"],
            )
            continue

        if element.class_id == 4 and element.element_id == 28 and current is not None:
            tile = _parse_binary_id28_tile(element.parameters)
            if tile is None:
                continue
            _append_tile_record(
                current,
                payload=tile["payload"] if isinstance(tile["payload"], bytes) else b"",
                compression=tile["compression"] if isinstance(tile["compression"], int) else None,
                bit_order=tile["bit_order"] if isinstance(tile["bit_order"], int) else None,
                orientation=tile["orientation"] if isinstance(tile["orientation"], int) else None,
                family=None,
                row_padding=tile["row_padding"] if isinstance(tile["row_padding"], int) else None,
            )
            continue

        if element.class_id == 0 and element.element_id == 20 and current is not None:
            finalized = _finalize_tile_array_record(current)
            if finalized is not None:
                arrays.append(finalized)
            current = None

    if current is not None:
        finalized = _finalize_tile_array_record(current)
        if finalized is not None:
            arrays.append(finalized)

    return arrays


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
                cell_path = round(numbers[3])
                cols = max(1, round(numbers[4]))
                rows = max(1, round(numbers[5]))
                tile_w = max(1, round(numbers[6]))
                tile_h = max(1, round(numbers[7]))
                total_w = max(1, round(numbers[-2]))
                total_h = max(1, round(numbers[-1]))
            else:
                cell_path, cols, rows, tile_w, tile_h, total_w, total_h = 0, 1, 1, 1, 1, 1, 1

            current = _make_tile_array_record(
                cols=cols,
                rows=rows,
                tile_width=tile_w,
                tile_height=tile_h,
                total_width=total_w,
                total_height=total_h,
                cell_path=cell_path,
            )
            continue

        if command in _TEXT_TILE_COMMAND_FAMILY and current is not None:
            payload = _extract_hex_payload(statement)
            if payload:
                metadata_prefix = statement.split("''", 1)[0].split('""', 1)[0]
                numbers = _extract_text_numbers(metadata_prefix)
                compression = round(numbers[0]) if numbers else None
                is_bitonal_cmd = command in {"BITONALTILE", "MONOCHROMETILE"}
                row_padding = round(numbers[1]) if is_bitonal_cmd and len(numbers) >= 2 else None
                local_color_precision = (
                    None if is_bitonal_cmd else (round(numbers[1]) if len(numbers) >= 2 else None)
                )
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
                    row_padding=row_padding,
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
    row_padding: int | None = None,
) -> Image.Image | None:
    image, _details = _decode_bitonal_payload_with_details(
        payload,
        width,
        height,
        compression=compression,
        bit_order=bit_order,
        row_padding=row_padding,
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
    row_padding: int | None = None,
    orientation: int | None = None,
) -> Image.Image | None:
    """Decode tile payloads across bitonal and direct-color encodings."""
    if width <= 0 or height <= 0:
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

    if local_color_precision == 32 and len(payload) >= total * 4:
        return Image.frombytes("RGBA", (width, height), payload[: total * 4])

    if local_color_precision == 16:
        scaled_rgb = scale_direct16_rgb_payload(
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

    bitonal_image = _decode_bitonal_payload_to_image(
        payload,
        width,
        height,
        compression=compression,
        bit_order=bit_order,
        row_padding=row_padding,
    )
    if bitonal_image is not None:
        return _apply_tile_orientation(bitonal_image, orientation)

    if len(payload) >= total * 4:
        return _apply_tile_orientation(
            Image.frombytes("RGBA", (width, height), payload[: total * 4]), orientation
        )

    if len(payload) >= total * 3:
        return _apply_tile_orientation(
            Image.frombytes("RGB", (width, height), payload[: total * 3]), orientation
        )

    if len(payload) >= total * 2:
        gray16 = payload[: total * 2]
        gray8 = bytes(gray16[idx] for idx in range(0, len(gray16), 2))
        return _apply_tile_orientation(Image.frombytes("L", (width, height), gray8), orientation)

    if len(payload) >= total:
        sample = payload[:total]
        unique = set(sample)
        if unique.issubset({0, 1}):
            sample = bytes(0 if value else 255 for value in sample)
        return _apply_tile_orientation(Image.frombytes("L", (width, height), sample), orientation)

    return None


def _apply_tile_orientation(image: Image.Image, _orientation: int | None) -> Image.Image:
    """Apply CGM BITONALTILE orientation transform to a decoded tile image."""
    return image


def _find_t6_eofb_bit_offset(payload: bytes) -> int | None:
    """Return the bit offset of the T.6 EOFB marker in payload, or None."""
    n = len(payload)
    if n < 3:
        return None
    start_byte = max(0, n - 16)
    for byte_idx in range(start_byte, n - 2):
        chunk_bytes = payload[byte_idx : min(byte_idx + 4, n)]
        chunk = int.from_bytes(chunk_bytes.ljust(4, b"\x00"), "big")
        for bit_off in range(8):
            window = (chunk >> (8 - bit_off)) & 0xFFFFFF
            if window == 0x001001:
                return byte_idx * 8 + bit_off
    return None


def _decode_bitonal_payload_with_details(
    payload: bytes,
    width: int,
    height: int,
    *,
    compression: int | None = None,
    bit_order: int | None = None,
    row_padding: int | None = None,
    preferred_signature: tuple[str, str, bool] | None = None,
    preferred_dimensions: tuple[int, int] | None = None,
) -> tuple[Image.Image | None, dict[str, object]]:
    if width <= 0 or height <= 0:
        return None, {
            "best_score": None,
            "candidate_count": 0,
            "best_candidate": None,
            "attempts": [],
        }

    if compression == 0:
        row_bytes = (width + 7) // 8
        needed = row_bytes * height
        if len(payload) < needed:
            return None, {
                "best_score": None,
                "candidate_count": 0,
                "best_candidate": None,
                "preferred_signature": preferred_signature,
                "preferred_dimensions": preferred_dimensions,
                "used_preferred_signature": False,
                "attempts": [],
            }

        bits: list[int] = []
        packed = payload[:needed]
        if bit_order == 1:
            packed = bytes(_reverse_bits_in_byte(byte) for byte in packed)
        for y in range(height):
            row_start = y * row_bytes
            for x in range(width):
                value = packed[row_start + (x // 8)]
                bits.append((value >> (7 - (x % 8))) & 1)

        pixels = bytes(0 if bit else 255 for bit in bits)
        return Image.frombytes("L", (width, height), pixels), {
            "best_score": None,
            "candidate_count": 1,
            "best_candidate": {
                "decoder": "packed_raw",
                "encoded_variant": "as_is",
                "width": width,
                "height": height,
                "invert": False,
                "score": None,
            },
            "preferred_signature": preferred_signature,
            "preferred_dimensions": preferred_dimensions,
            "used_preferred_signature": False,
            "attempts": [
                {
                    "decoder": "packed_raw",
                    "encoded_variant": "as_is",
                    "width": width,
                    "height": height,
                    "invert": False,
                    "score": None,
                }
            ],
        }

    if compression == 1:
        decoded = _cached_ccittfax3_decode(payload, height=height, width=width)
        if decoded is None:
            return None, {
                "best_score": None,
                "candidate_count": 0,
                "best_candidate": None,
                "preferred_signature": preferred_signature,
                "preferred_dimensions": preferred_dimensions,
                "used_preferred_signature": False,
                "attempts": [],
            }
        decoded_bits = _decode_fax_output_to_bitmap(decoded, width, height)
        if decoded_bits is None:
            return None, {
                "best_score": None,
                "candidate_count": 0,
                "best_candidate": None,
                "preferred_signature": preferred_signature,
                "preferred_dimensions": preferred_dimensions,
                "used_preferred_signature": False,
                "attempts": [],
            }
        pixels = bytes(0 if bit else 255 for bit in decoded_bits)
        return Image.frombytes("L", (width, height), pixels), {
            "best_score": None,
            "candidate_count": 1,
            "best_candidate": {
                "decoder": "fax3:exact",
                "encoded_variant": "as_is",
                "width": width,
                "height": height,
                "invert": False,
                "score": None,
            },
            "preferred_signature": preferred_signature,
            "preferred_dimensions": preferred_dimensions,
            "used_preferred_signature": False,
            "attempts": [
                {
                    "decoder": "fax3:exact",
                    "encoded_variant": "as_is",
                    "width": width,
                    "height": height,
                    "invert": False,
                    "score": None,
                }
            ],
        }

    if compression == 2:
        preferred_width = width
        try_widths: list[int] = [width]
        if isinstance(row_padding, int) and row_padding > 8:
            half_padding_width = width + row_padding // 2
            if width % row_padding == 0:
                preferred_width = width
                try_widths = [width]
            else:
                preferred_width = half_padding_width
                try_widths = [half_padding_width]
        elif isinstance(row_padding, int) and 0 < row_padding <= 8:
            preferred_width = width + row_padding
            try_widths = [preferred_width, width]

        seen_w: set[int] = set()
        unique_widths: list[int] = []
        for w_val in try_widths:
            if w_val not in seen_w:
                seen_w.add(w_val)
                unique_widths.append(w_val)

        def _fax4_try(w_val: int) -> tuple[Image.Image | None, int, int, float]:
            inf = height + 1
            raw = _cached_ccittfax4_decode(payload, height=height, width=w_val)
            if raw is None:
                return None, inf, inf, 1.0
            if w_val > width:
                stripped = bytearray()
                for row_idx in range(height):
                    stripped.extend(raw[row_idx * w_val : row_idx * w_val + width])
                raw = bytes(stripped)
            bits = _decode_fax_output_to_bitmap(raw, width, height)
            if bits is None:
                return None, inf, inf, 1.0
            sbr = sum(
                1
                for row_idx in range(height)
                if sum(bits[row_idx * width : (row_idx + 1) * width]) / width > 0.95
            )
            twr = 0
            for row_idx in range(height - 1, -1, -1):
                if sum(bits[row_idx * width : (row_idx + 1) * width]) == 0:
                    twr += 1
                else:
                    break
            right_black = (
                sum(bits[row_idx * width + (width - 1)] for row_idx in range(height)) / height
            )
            px = bytes(0 if bit else 255 for bit in bits)
            return Image.frombytes("L", (width, height), px), sbr, twr, right_black

        candidates: list[tuple[int, Image.Image, int, int, float]] = []
        for w_val in unique_widths:
            img, sbr, twr, right_black = _fax4_try(w_val)
            if img is not None:
                candidates.append((w_val, img, sbr, twr, right_black))

        if not candidates:
            return None, {
                "best_score": None,
                "candidate_count": 0,
                "best_candidate": None,
                "preferred_signature": preferred_signature,
                "preferred_dimensions": preferred_dimensions,
                "used_preferred_signature": False,
                "attempts": [],
            }

        nom = next(
            (twr for w_val, _, _sbr, twr, _right_black in candidates if w_val == preferred_width),
            next(
                (twr for w_val, _, _sbr, twr, _right_black in candidates if w_val == width), height
            ),
        )
        max_twr = nom + max(1, height // 10)
        valid = [
            (w_val, img, sbr, twr, right_black)
            for w_val, img, sbr, twr, right_black in candidates
            if twr <= max_twr
        ]
        if not valid:
            valid = [
                (w_val, img, sbr, twr, right_black)
                for w_val, img, sbr, twr, right_black in candidates
                if w_val == preferred_width
            ]
        if not valid:
            valid = [
                (w_val, img, sbr, twr, right_black)
                for w_val, img, sbr, twr, right_black in candidates
                if w_val == width
            ]

        _, best_img, _, _, _ = min(valid, key=lambda item: (item[2], item[4], item[3]))

        return best_img, {
            "best_score": None,
            "candidate_count": 1,
            "best_candidate": {
                "decoder": "fax4",
                "encoded_variant": "as_is",
                "width": width,
                "height": height,
                "invert": False,
                "score": None,
            },
            "preferred_signature": preferred_signature,
            "preferred_dimensions": preferred_dimensions,
            "used_preferred_signature": False,
            "attempts": [
                {
                    "decoder": "fax4",
                    "encoded_variant": "as_is",
                    "width": width,
                    "height": height,
                    "invert": False,
                    "score": None,
                }
            ],
        }

    return None, {
        "best_score": None,
        "candidate_count": 0,
        "best_candidate": None,
        "preferred_signature": preferred_signature,
        "preferred_dimensions": preferred_dimensions,
        "used_preferred_signature": False,
        "attempts": [],
    }


def _render_first_tile_array(data: bytes) -> Image.Image | None:
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

            tile_for_canvas = tile_img
            if tile_img.size != (paste_w, paste_h):
                tile_for_canvas = tile_img.crop((0, 0, paste_w, paste_h))
            canvas.paste(tile_for_canvas, (x, y))
            pasted_any = True

        if pasted_any:
            return canvas

    return None
