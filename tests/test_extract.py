from __future__ import annotations

__author__ = "Kestin Goforth"
__copyright__ = "Copyright 2026"
__license__ = "BSD-3-Clause"

import json
import struct
from typing import TYPE_CHECKING

from cgm.extract import (
    extract_data_json_from_bytes,
    extract_raw_images_from_bytes,
    extract_raw_images_to_directory,
    extract_vector_svg_from_bytes,
)

if TYPE_CHECKING:
    from pathlib import Path


def _header(class_id: int, element_id: int, length: int) -> bytes:
    value = (class_id << 12) | (element_id << 5) | length
    return value.to_bytes(2, "big")


def _encode_f32(value: float) -> bytes:
    return struct.pack(">f", value)


def test_extract_raw_images_from_bytes_reads_cell_array_payload() -> None:
    params = (
        (b"\x00" * 12)
        + (2).to_bytes(2, "big")
        + (3).to_bytes(2, "big")
        + (b"\x00" * 4)
        + b"PAYLOAD"
    )
    data = _header(4, 9, len(params)) + params

    images = extract_raw_images_from_bytes(data)

    assert len(images) == 1
    assert images[0].width == 2
    assert images[0].height == 3
    assert images[0].payload == b"PAYLOAD"


def test_extract_vector_svg_from_bytes_emits_svg_document() -> None:
    points = _encode_f32(0.0) + _encode_f32(0.0) + _encode_f32(10.0) + _encode_f32(10.0)
    data = _header(4, 1, len(points)) + points

    svg = extract_vector_svg_from_bytes(data)

    assert "<svg" in svg
    assert "<polyline" in svg


def test_extract_data_json_from_bytes_contains_core_sections() -> None:
    params = b"\x00" * 4
    data = _header(4, 1, len(params)) + params

    snapshot = json.loads(extract_data_json_from_bytes(data))

    assert snapshot["byte_length"] == len(data)
    assert "elements" in snapshot
    assert "element_histogram" in snapshot
    assert "vector_svg" in snapshot


def test_extract_raw_images_to_directory_writes_payload_files(tmp_path: Path) -> None:
    params = (b"\x00" * 12) + (1).to_bytes(2, "big") + (1).to_bytes(2, "big") + (b"\x00" * 4) + b"X"
    data = _header(4, 9, len(params)) + params
    src = tmp_path / "sample.cgm"
    src.write_bytes(data)

    written = extract_raw_images_to_directory(src, tmp_path, stem="payload")

    assert len(written) == 1
    assert written[0].name == "payload_0000.bin"
    assert written[0].read_bytes() == b"X"
