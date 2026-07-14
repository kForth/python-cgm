"""Raster decoding and element-29 diagnostics helpers used by SVG rendering."""

from __future__ import annotations

import base64
import io
import logging
import math
from collections import Counter
from typing import Any

import imagecodecs
from PIL import Image

from .common import _decode_restricted_text

log = logging.getLogger("cgm.raster")


def _ascii_printable_ratio(data: bytes) -> float:
    if not data:
        return 0.0
    printable = sum(32 <= byte <= 126 for byte in data)
    return printable / len(data)


def _shannon_entropy(data: bytes) -> float:
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
    runs: list[dict[str, object]] = []
    idx = 0
    while idx < len(data) and len(runs) < limit:
        if 32 <= data[idx] <= 126:
            end = idx + 1
            while end < len(data) and 32 <= data[end] <= 126:
                end += 1
            if end - idx >= min_len:
                preview = data[idx:end][:48].decode("ascii", errors="ignore")
                runs.append({"offset": idx, "length": end - idx, "preview": preview})
            idx = end
        else:
            idx += 1
    return runs


def _leading_run_length(data: bytes, *, value: int) -> int:
    count = 0
    for byte in data:
        if byte != value:
            break
        count += 1
    return count


def _analyze_element29_payload(parameters: bytes) -> dict[str, object]:
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


def _bitmap_to_png_data_uri(bits: list[int], width: int, height: int) -> str | None:
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
    candidate_sizes = [(768, 1099), (767, 1099), (768, 1100), (576, 768), (575, 767)]

    best_score = 1e9
    best_bits: list[int] | None = None
    best_size: tuple[int, int] | None = None
    best_black_ratio = -1.0

    for offset in candidate_offsets:
        if offset >= len(parameters):
            continue
        payload = parameters[offset:]
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
