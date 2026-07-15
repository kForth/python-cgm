from __future__ import annotations

__author__ = "Kestin Goforth"
__copyright__ = "Copyright 2026"
__license__ = "BSD-3-Clause"

import inspect
import json
import struct
from typing import TYPE_CHECKING

import pytest

import cgm.extract as extract_module
from cgm.extract import (
    extract_data_json_from_bytes,
    extract_final_image_and_hotspots,
    extract_hotspots_from_bytes,
    extract_raw_images_from_bytes,
    extract_raw_images_to_directory,
    extract_rendered_images_to_directory,
    extract_vector_svg_from_bytes,
)

if TYPE_CHECKING:
    from pathlib import Path


def _header(class_id: int, element_id: int, length: int) -> bytes:
    value = (class_id << 12) | (element_id << 5) | length
    return value.to_bytes(2, "big")


def _encode_f32(value: float) -> bytes:
    return struct.pack(">f", value)


def _apd_property(key: bytes, value: bytes) -> bytes:
    return bytes([len(key)]) + key + bytes([len(value)]) + value


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


def test_extract_vector_svg_from_bytes_renders_restricted_text() -> None:
    vdc_extent = (
        (0).to_bytes(2, "big")
        + (0).to_bytes(2, "big")
        + (272).to_bytes(2, "big")
        + (394).to_bytes(2, "big")
    )
    extent_element = _header(2, 6, len(vdc_extent)) + vdc_extent

    # class-4/id-29 Restricted Text: box_w, box_h, anchor_x, anchor_y, string.
    text = b"HELLO"
    restricted_params = (
        (120).to_bytes(2, "big")
        + (20).to_bytes(2, "big")
        + (30).to_bytes(2, "big")
        + (40).to_bytes(2, "big")
        + bytes([len(text)])
        + text
    )
    restricted_element = _header(4, 29, len(restricted_params)) + restricted_params

    svg = extract_vector_svg_from_bytes(extent_element + restricted_element)

    assert "HELLO" in svg
    assert 'textLength="120.000"' in svg


def test_extract_data_json_from_bytes_includes_element29_analysis() -> None:
    payload = b"\x00\x02\x00\x00\x00\x01\x00" + (b"\xff" * 12) + b"ABCD"
    element29 = _header(4, 29, len(payload)) + payload

    snapshot = json.loads(extract_data_json_from_bytes(element29))

    assert "element29_analysis" in snapshot
    assert len(snapshot["element29_analysis"]) == 1
    analysis = snapshot["element29_analysis"][0]
    assert analysis["length"] == len(payload)
    assert "entropy_bits_per_byte" in analysis


def test_extract_hotspots_from_bytes_parses_name_and_region() -> None:
    begin_name = b"SPOT1"
    begin_apd = bytes([len(begin_name)]) + begin_name + b"\x00\x00"

    name_prop = _apd_property(b"name", b"A_ZONE")
    region_value = (
        b"\x00\x0b\x00\x01\x00\x01\x00\x10\x00\x04"
        + (248).to_bytes(2, "big")
        + (508).to_bytes(2, "big")
        + (267).to_bytes(2, "big")
        + (527).to_bytes(2, "big")
    )
    region_prop = _apd_property(b"region", region_value)

    data = (
        _header(0, 21, len(begin_apd))
        + begin_apd
        + _header(9, 1, len(name_prop))
        + name_prop
        + _header(9, 1, len(region_prop))
        + region_prop
        + _header(0, 22, 0)
    )

    hotspots = extract_hotspots_from_bytes(data)

    assert len(hotspots) == 1
    hotspot = hotspots[0]
    assert hotspot.source_tag == "SPOT1"
    assert hotspot.name == "A_ZONE"
    assert hotspot.x_min == 248
    assert hotspot.y_min == 508
    assert hotspot.x_max == 267
    assert hotspot.y_max == 527


def test_extract_data_json_from_bytes_includes_hotspots() -> None:
    begin_name = b"SPOT2"
    begin_apd = bytes([len(begin_name)]) + begin_name + b"\x00\x00"
    name_prop = _apd_property(b"name", b"B_ZONE")
    region_value = (
        b"\x00\x0b\x00\x01\x00\x01\x00\x10\x00\x04"
        + (10).to_bytes(2, "big")
        + (20).to_bytes(2, "big")
        + (30).to_bytes(2, "big")
        + (40).to_bytes(2, "big")
    )
    region_prop = _apd_property(b"region", region_value)

    data = (
        _header(0, 21, len(begin_apd))
        + begin_apd
        + _header(9, 1, len(name_prop))
        + name_prop
        + _header(9, 1, len(region_prop))
        + region_prop
        + _header(0, 22, 0)
    )

    snapshot = json.loads(extract_data_json_from_bytes(data))

    assert "hotspots" in snapshot
    assert len(snapshot["hotspots"]) == 1
    assert snapshot["hotspots"][0]["name"] == "B_ZONE"


def test_extract_final_image_and_hotspots_returns_svg_and_hotspots(tmp_path: Path) -> None:
    begin_name = b"SPOT1"
    begin_apd = bytes([len(begin_name)]) + begin_name + b"\x00\x00"
    name_prop = _apd_property(b"name", b"A_ZONE")
    region_value = (
        b"\x00\x0b\x00\x01\x00\x01\x00\x10\x00\x04"
        + (10).to_bytes(2, "big")
        + (20).to_bytes(2, "big")
        + (30).to_bytes(2, "big")
        + (40).to_bytes(2, "big")
    )
    region_prop = _apd_property(b"region", region_value)
    points = _encode_f32(0.0) + _encode_f32(0.0) + _encode_f32(10.0) + _encode_f32(10.0)
    data = (
        _header(4, 1, len(points))
        + points
        + _header(0, 21, len(begin_apd))
        + begin_apd
        + _header(9, 1, len(name_prop))
        + name_prop
        + _header(9, 1, len(region_prop))
        + region_prop
        + _header(0, 22, 0)
    )
    src = tmp_path / "sample.cgm"
    src.write_bytes(data)

    result = extract_final_image_and_hotspots(src)

    assert isinstance(result["image"], str)
    assert "<svg" in result["image"]
    assert isinstance(result["hotspots"], list)
    assert result["hotspots"]
    assert result["hotspots"][0]["name"] == "A_ZONE"


def test_extract_final_image_and_hotspots_rejects_image_format_kwarg(tmp_path: Path) -> None:
    params = (
        (b"\x00" * 12) + (1).to_bytes(2, "big") + (1).to_bytes(2, "big") + (b"\x00" * 4) + b"\x01"
    )
    src = tmp_path / "sample.cgm"
    src.write_bytes(_header(4, 9, len(params)) + params)

    signature = inspect.signature(extract_final_image_and_hotspots)
    assert "image_format" not in signature.parameters


def test_extract_vector_svg_from_clear_text_cgm() -> None:
    data = b'VDCEXT 0 0 100 100; LINE 0 0 10 10; TEXT 12 34 "NOTE";'

    svg = extract_vector_svg_from_bytes(data)

    assert "<svg" in svg
    assert "<polyline" in svg
    assert "NOTE" in svg


def test_extract_vector_svg_from_bytes_supports_additional_primitives() -> None:
    disjoint = _encode_f32(0.0) + _encode_f32(0.0) + _encode_f32(10.0) + _encode_f32(10.0)
    markers = _encode_f32(5.0) + _encode_f32(5.0)
    polygon_set = (
        _encode_f32(10.0)
        + _encode_f32(10.0)
        + _encode_f32(20.0)
        + _encode_f32(10.0)
        + _encode_f32(15.0)
        + _encode_f32(20.0)
    )
    rectangle = _encode_f32(40.0) + _encode_f32(40.0) + _encode_f32(60.0) + _encode_f32(70.0)
    circle = _encode_f32(80.0) + _encode_f32(80.0) + _encode_f32(5.0)
    ellipse = (
        _encode_f32(120.0)
        + _encode_f32(120.0)
        + _encode_f32(135.0)
        + _encode_f32(120.0)
        + _encode_f32(120.0)
        + _encode_f32(140.0)
    )

    data = (
        _header(4, 2, len(disjoint))
        + disjoint
        + _header(4, 3, len(markers))
        + markers
        + _header(4, 8, len(polygon_set))
        + polygon_set
        + _header(4, 11, len(rectangle))
        + rectangle
        + _header(4, 12, len(circle))
        + circle
        + _header(4, 17, len(ellipse))
        + ellipse
    )

    svg = extract_vector_svg_from_bytes(data)

    assert "<polyline" in svg
    assert "<polygon" in svg
    assert "<rect" in svg
    assert "<ellipse" in svg
    assert svg.count("<circle") >= 2


def test_extract_vector_svg_from_bytes_supports_binary_integer_polyline() -> None:
    # Binary streams often encode VDC coordinates as 16-bit signed integers.
    points_i16 = (
        (0).to_bytes(2, "big", signed=True)
        + (0).to_bytes(2, "big", signed=True)
        + (100).to_bytes(2, "big", signed=True)
        + (100).to_bytes(2, "big", signed=True)
    )
    data = _header(4, 1, len(points_i16)) + points_i16

    svg = extract_vector_svg_from_bytes(data)

    assert "<svg" in svg
    assert "<polyline" in svg


def test_extract_vector_svg_from_bytes_supports_extended_binary_arc_family() -> None:
    segment = _encode_f32(0.0) + _encode_f32(0.0) + _encode_f32(10.0) + _encode_f32(10.0)
    data = b"".join(_header(4, element_id, len(segment)) + segment for element_id in range(20, 29))

    svg = extract_vector_svg_from_bytes(data)

    assert "<svg" in svg
    assert "<polyline" in svg
    assert "unsupported drawing primitives" not in svg


def test_extract_vector_svg_from_bytes_renders_binary_gdp_with_other_primitives() -> None:
    polyline = _encode_f32(0.0) + _encode_f32(0.0) + _encode_f32(10.0) + _encode_f32(10.0)
    gdp_polyline = _encode_f32(20.0) + _encode_f32(20.0) + _encode_f32(30.0) + _encode_f32(30.0)
    data = (
        _header(4, 1, len(polyline))
        + polyline
        + _header(4, 10, len(gdp_polyline))
        + gdp_polyline
        + _header(4, 26, len(gdp_polyline))
        + gdp_polyline
    )

    svg = extract_vector_svg_from_bytes(data)

    assert "<svg" in svg
    assert svg.count("<polyline") >= 3


def test_extract_raw_images_from_clear_text_cellarray() -> None:
    data = b"CELLARRAY 0 0 1 0 0 1 2 2 8 1 2 3 4;"

    images = extract_raw_images_from_bytes(data)

    assert len(images) == 1
    assert images[0].width == 2
    assert images[0].height == 2
    assert images[0].payload == b"\x01\x02\x03\x04"


def test_extract_hotspots_from_clear_text_application_data() -> None:
    data = b'BEGAPS "SPOT_TEXT"; APD "name" "ZONE_TEXT"; APD "region" "10 20 30 40"; ENDAPS;'

    hotspots = extract_hotspots_from_bytes(data)

    assert len(hotspots) == 1
    assert hotspots[0].source_tag == "SPOT_TEXT"
    assert hotspots[0].name == "ZONE_TEXT"
    assert hotspots[0].x_min == 10
    assert hotspots[0].y_min == 20
    assert hotspots[0].x_max == 30
    assert hotspots[0].y_max == 40


def test_extract_hotspots_from_bytes_aps_geometry_fallback() -> None:
    begin_name = b"AUTOID_1"
    begin_apd = bytes([len(begin_name)]) + begin_name + b"\x00"
    polygon_points = (
        _encode_f32(10.0)
        + _encode_f32(20.0)
        + _encode_f32(30.0)
        + _encode_f32(20.0)
        + _encode_f32(30.0)
        + _encode_f32(40.0)
    )
    restricted_text = (
        (20).to_bytes(2, "big")
        + (10).to_bytes(2, "big")
        + (12).to_bytes(2, "big")
        + (24).to_bytes(2, "big")
        + bytes([3])
        + b"TAG"
    )

    data = (
        _header(0, 21, len(begin_apd))
        + begin_apd
        + _header(4, 7, len(polygon_points))
        + polygon_points
        + _header(4, 29, len(restricted_text))
        + restricted_text
        + _header(0, 22, 0)
    )

    hotspots = extract_hotspots_from_bytes(data)

    assert len(hotspots) == 1
    hotspot = hotspots[0]
    assert hotspot.source_tag == "AUTOID_1"
    assert hotspot.name == "AUTOID_1"
    assert hotspot.x_min == 10
    assert hotspot.y_min == 20
    assert hotspot.x_max == 32
    assert hotspot.y_max == 40


def test_extract_hotspots_from_gr_77775_contains_autoid_regions() -> None:
    from pathlib import Path  # noqa: PLC0415

    root = Path(__file__).resolve().parents[1]
    candidates = [
        root / "sample" / "GR-77775.cgm",
        root / "test_files" / "GR-77775.cgm",
    ]
    source = next((path for path in candidates if path.exists()), candidates[0])
    hotspots = extract_hotspots_from_bytes(source.read_bytes())

    assert hotspots
    tags = {item.source_tag for item in hotspots}
    assert "AUTOID_1" in tags


def test_element29_binary_fallback_rejects_low_confidence_gr_283383() -> None:
    from pathlib import Path  # noqa: PLC0415

    root = Path(__file__).resolve().parents[1]
    candidates = [
        root / "sample" / "GR-283383.cgm",
        root / "test_files" / "GR-283383.cgm",
    ]
    source = next((path for path in candidates if path.exists()), candidates[0])

    payload = None
    for element in extract_module.iter_elements(source.read_bytes()):
        if element.class_id == 4 and element.element_id == 29:
            payload = element.parameters
            break

    assert payload is not None
    decode_element29 = extract_module._decode_element29_binary_raster
    assert decode_element29(payload) is None


def test_gr_283383_skips_unsupported_fallback_rendering() -> None:
    from pathlib import Path  # noqa: PLC0415

    root = Path(__file__).resolve().parents[1]
    candidates = [
        root / "sample" / "GR-283383.cgm",
        root / "test_files" / "GR-283383.cgm",
    ]
    source = next((path for path in candidates if path.exists()), candidates[0])

    svg = extract_vector_svg_from_bytes(source.read_bytes())

    assert "<image " not in svg
    assert "unsupported drawing primitives" not in svg
    assert "POSID_" not in svg


def test_gr_78946_renders_binary_text_and_arcs() -> None:
    from pathlib import Path  # noqa: PLC0415

    root = Path(__file__).resolve().parents[1]
    candidates = [
        root / "sample" / "GR-78946.cgm",
        root / "test_files" / "GR-78946.cgm",
    ]
    source = next((path for path in candidates if path.exists()), candidates[0])

    svg = extract_vector_svg_from_bytes(source.read_bytes())

    assert "<text " in svg
    assert "<polyline" in svg or "<polygon" in svg
    assert "unsupported drawing primitives" not in svg


def test_extract_raw_images_from_clear_text_bitonal_tiles() -> None:
    hex_payload = "0123456789ABCDEFFEDCBA9876543210"
    data = (
        "BEGTILEARRAY 0 0 0 0 1 1 2 2 0 0 0 0 2 2; "
        f"BITONALTILE 2 16 0 1 '' {hex_payload}; "
        "ENDTILEARRAY;"
    ).encode("ascii")

    images = extract_raw_images_from_bytes(data)

    assert len(images) == 1
    assert images[0].width == 2
    assert images[0].height == 2
    assert images[0].payload == bytes.fromhex(hex_payload)


def test_extract_rendered_images_to_directory_writes_svg(tmp_path: Path) -> None:
    params = (
        (b"\x00" * 12) + (1).to_bytes(2, "big") + (1).to_bytes(2, "big") + (b"\x00" * 4) + b"\x01"
    )
    data = _header(4, 9, len(params)) + params
    src = tmp_path / "sample.cgm"
    src.write_bytes(data)

    written = extract_rendered_images_to_directory(src, tmp_path, stem="rendered")

    assert written
    assert any(path.suffix == ".svg" for path in written)


def test_extract_rendered_images_to_directory_stitches_tile_arrays(tmp_path: Path) -> None:
    src = tmp_path / "tiles.cgm"
    hex_payload = "0123456789ABCDEFFEDCBA9876543210"
    src.write_text(
        "BEGTILEARRAY 0 0 0 0 1 1 1 1 0 0 0 0 1 1; "
        f"BITONALTILE 2 16 0 1 '' {hex_payload}; "
        "ENDTILEARRAY;",
        encoding="ascii",
    )

    written = extract_rendered_images_to_directory(src, tmp_path, stem="tiles")

    svg_path = tmp_path / "tiles_0000.svg"
    assert svg_path in written
    svg = svg_path.read_text(encoding="utf-8")
    assert "<svg" in svg
    assert "<image" in svg


def test_extract_rendered_images_to_directory_writes_decode_report(tmp_path: Path) -> None:
    src = tmp_path / "tiles.cgm"
    hex_payload = "0123456789ABCDEFFEDCBA9876543210"
    src.write_text(
        "BEGTILEARRAY 0 0 0 0 1 1 1 1 0 0 0 0 1 1; "
        f"BITONALTILE 2 16 0 1 '' {hex_payload}; "
        "ENDTILEARRAY;",
        encoding="ascii",
    )

    written = extract_rendered_images_to_directory(src, tmp_path, stem="tiles", debug_report=True)

    report_path = tmp_path / "tiles_decode_report.json"
    assert report_path in written
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert "arrays" in report
    assert report["arrays"]
    first_array = report["arrays"][0]
    assert first_array["tile_count"] == 1


def test_score_bitmap_is_polarity_agnostic() -> None:
    width = 8
    height = 8
    bits = [
        1,
        1,
        1,
        0,
        0,
        0,
        1,
        0,
    ] * height
    inverted = [1 - bit for bit in bits]
    score_bitmap = extract_module._score_bitmap

    score = score_bitmap(bits, width, height)
    inverted_score = score_bitmap(inverted, width, height)

    assert score == pytest.approx(inverted_score)


def test_score_bitmap_rejects_nearly_single_color_regardless_of_polarity() -> None:
    width = 100
    height = 100
    mostly_black = [1] * (width * height)
    mostly_black[0] = 0
    mostly_white = [0] * (width * height)
    mostly_white[0] = 1
    score_bitmap = extract_module._score_bitmap

    black_score = score_bitmap(mostly_black, width, height)
    white_score = score_bitmap(mostly_white, width, height)

    assert black_score >= 1e8
    assert white_score >= 1e8
