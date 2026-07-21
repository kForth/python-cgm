from __future__ import annotations

__author__ = "Kestin Goforth"
__copyright__ = "Copyright 2026"
__license__ = "BSD-3-Clause"

import base64
import importlib
import inspect
import io
import json
import re
import struct
from collections import Counter
from typing import TYPE_CHECKING

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
from cgm.extract.core import decode_restricted_text
from cgm.extract.svg import _parse_element29_raster_prefix, _render_first_element29_raster
from cgm.parser import iter_elements

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
    assert images[0].local_color_precision == 0
    assert images[0].cell_representation_mode == 0
    assert images[0].payload == b"PAYLOAD"


def test_extract_raw_images_from_clear_text_cellarray_preserves_metadata() -> None:
    data = b"CELLARRAY 0 0 1 0 0 1 2 2 8 1 2 3 4;"

    images = extract_raw_images_from_bytes(data)

    assert len(images) == 1
    assert images[0].width == 2
    assert images[0].height == 2
    assert images[0].local_color_precision == 8
    assert images[0].cell_representation_mode == 0


def test_render_raw_image_payload_prefers_indexed_when_precision_is_8bit() -> None:
    raw = extract_module.RawImage(
        index=0,
        element_offset=0,
        payload=bytes([0, 1, 2, 3]),
        width=2,
        height=2,
        local_color_precision=8,
        cell_representation_mode=0,
    )

    image = extract_module._render_raw_image_payload(raw)

    assert image is not None
    assert image.mode == "P"
    assert image.size == (2, 2)


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


def test_extract_color_table_handles_16bit_components() -> None:
    data = bytearray()
    color_table = bytes([0, 2]) + bytes.fromhex("ffff00008000")
    data += _header(5, 34, len(color_table)) + color_table

    table = extract_module._extract_color_table(bytes(data))

    assert table.get(2) == (255, 0, 128)


def test_extract_color_value_extent_from_binary_and_clear_text() -> None:
    binary = _header(1, 10, 12) + b"\x00\x00\x00\x00\x00\x00\x0f\xff\x0f\xff\x0f\xff"
    clear_text = b"COLRVALUEEXT 0 0 0 4095 4095 4095;"

    binary_extent = extract_module._extract_color_value_extent(binary)
    clear_text_extent = extract_module._extract_color_value_extent(clear_text)

    assert binary_extent == (0, 0, 0, 4095, 4095, 4095)
    assert clear_text_extent == (0, 0, 0, 4095, 4095, 4095)


def test_extract_vector_svg_from_clear_text_respects_transparency_mode() -> None:
    svg_off = extract_vector_svg_from_bytes(b"TRANSPARENCY OFF; LINE 0 0 10 10;")
    svg_on = extract_vector_svg_from_bytes(b"TRANSPARENCY ON; LINE 0 0 10 10;")

    assert 'fill="#ffffff"' in svg_off
    assert 'fill="#ffffff"' not in svg_on


def test_extract_vector_svg_from_bytes_respects_binary_clip_settings() -> None:
    clip_rect = _header(3, 5, 8) + b"\x00\x02\x00\x02\x00\x08\x00\x08"
    clip_on = _header(3, 6, 1) + b"\x01" + b"\x00"
    line = (
        _header(4, 1, 16)
        + _encode_f32(0.0)
        + _encode_f32(0.0)
        + _encode_f32(10.0)
        + _encode_f32(10.0)
    )

    svg = extract_vector_svg_from_bytes(clip_rect + clip_on + line)

    assert '<clipPath id="cgmClip0">' in svg
    assert 'clip-path="url(#cgmClip0)"' in svg


def test_extract_vector_svg_from_clear_text_respects_clip_commands() -> None:
    svg_on = extract_vector_svg_from_bytes(b"CLIPRECT 2 2 8 8; CLIPIND ON; LINE 0 0 10 10;")
    svg_off = extract_vector_svg_from_bytes(b"CLIPRECT 2 2 8 8; CLIPIND OFF; LINE 0 0 10 10;")

    assert '<clipPath id="cgmClip0">' in svg_on
    assert 'clip-path="url(#cgmClip0)"' in svg_on
    assert '<clipPath id="cgmClip0">' not in svg_off


def test_extract_vector_svg_from_bytes_supports_binary_integer_polyline() -> None:
    # Integer coordinate decode now requires explicit VDC descriptors.
    vdc_type_integer = _header(1, 3, 2) + (0).to_bytes(2, "big")
    vdc_integer_precision = _header(1, 11, 2) + (16).to_bytes(2, "big")
    points_i16 = (
        (0).to_bytes(2, "big", signed=True)
        + (0).to_bytes(2, "big", signed=True)
        + (100).to_bytes(2, "big", signed=True)
        + (100).to_bytes(2, "big", signed=True)
    )
    data = vdc_type_integer + vdc_integer_precision + _header(4, 1, len(points_i16)) + points_i16

    svg = extract_vector_svg_from_bytes(data)

    assert "<svg" in svg
    assert "<polyline" in svg


def test_extract_vector_svg_from_bytes_supports_binary_integer32_polyline() -> None:
    vdc_type_integer = _header(1, 3, 2) + (0).to_bytes(2, "big")
    vdc_integer_precision = _header(1, 11, 2) + (32).to_bytes(2, "big")
    points_i32 = (
        (2_500_000).to_bytes(4, "big", signed=True)
        + (2_500_000).to_bytes(4, "big", signed=True)
        + (2_600_000).to_bytes(4, "big", signed=True)
        + (2_600_000).to_bytes(4, "big", signed=True)
    )
    extent = (
        _encode_f32(2_400_000.0)
        + _encode_f32(2_400_000.0)
        + _encode_f32(2_700_000.0)
        + _encode_f32(2_700_000.0)
    )
    data = (
        vdc_type_integer
        + vdc_integer_precision
        + _header(2, 6, len(extent))
        + extent
        + _header(4, 1, len(points_i32))
        + points_i32
    )

    svg = extract_vector_svg_from_bytes(data)

    assert "<svg" in svg
    assert '<polyline points="2500000.000,2600000.000 2600000.000,2500000.000"' in svg


def test_extract_vector_svg_from_bytes_uses_vdc_integer_descriptor() -> None:
    vdc_type_integer = _header(1, 3, 2) + (0).to_bytes(2, "big")
    vdc_integer_precision = _header(1, 11, 2) + (16).to_bytes(2, "big")
    points_i16 = (
        (0).to_bytes(2, "big", signed=True)
        + (0).to_bytes(2, "big", signed=True)
        + (100).to_bytes(2, "big", signed=True)
        + (100).to_bytes(2, "big", signed=True)
    )
    data = vdc_type_integer + vdc_integer_precision + _header(4, 1, len(points_i16)) + points_i16

    svg = extract_vector_svg_from_bytes(data)

    assert '<polyline points="0.000,100.000 100.000,0.000"' in svg


def test_extract_vector_svg_from_bytes_uses_vdc_real_descriptor() -> None:
    vdc_type_real = _header(1, 3, 2) + (1).to_bytes(2, "big")
    # Minimal real-precision hint used by strict descriptor path.
    vdc_real_precision = _header(1, 12, 2) + (32).to_bytes(2, "big")
    points = _encode_f32(1.0) + _encode_f32(1.0) + _encode_f32(2.0) + _encode_f32(2.0)
    data = vdc_type_real + vdc_real_precision + _header(4, 1, len(points)) + points

    svg = extract_vector_svg_from_bytes(data)

    assert '<polyline points="1.000,2.000 2.000,1.000"' in svg


def test_extract_raw_images_from_clear_text_cellarray() -> None:
    data = b"CELLARRAY 0 0 1 0 0 1 2 2 8 1 2 3 4;"

    images = extract_raw_images_from_bytes(data)

    assert len(images) == 1
    assert images[0].width == 2
    assert images[0].height == 2
    assert images[0].payload == b"\x01\x02\x03\x04"


def test_parse_text_tile_arrays_preserves_precision_and_representation_mode() -> None:
    payload = "FF0000" * 12
    data = (
        "BEGTILEARRAY 0 0 0 0 1 1 1 1 0 0 0 0 1 1; "
        f"COLORTILE 2 24 0 1 1 '' {payload}; "
        "ENDTILEARRAY;"
    ).encode("ascii")

    arrays = extract_module._parse_text_tile_arrays(data)

    assert len(arrays) == 1
    tiles = arrays[0]["tiles"]
    assert isinstance(tiles, list)
    first_tile = tiles[0]
    assert isinstance(first_tile, dict)
    assert first_tile.get("local_color_precision") == 24
    assert first_tile.get("cell_representation_mode") == 1


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


def test_gr_283383_data_snapshot_defaults_to_non_heuristic_element29_analysis() -> None:
    from pathlib import Path  # noqa: PLC0415

    root = Path(__file__).resolve().parents[1]
    candidates = [
        root / "sample" / "GR-283383.cgm",
        root / "test_files" / "GR-283383.cgm",
    ]
    source = next((path for path in candidates if path.exists()), candidates[0])

    snapshot = json.loads(extract_data_json_from_bytes(source.read_bytes()))
    element29 = snapshot["element29_analysis"][0]

    assert "decode_candidates" in element29
    assert element29["decode_candidates"] == []
    assert element29["has_plausible_decode"] is False


def test_gr_283383_default_svg_does_not_use_heuristic_class29_fallback() -> None:
    from pathlib import Path  # noqa: PLC0415

    root = Path(__file__).resolve().parents[1]
    candidates = [
        root / "sample" / "GR-283383.cgm",
        root / "test_files" / "GR-283383.cgm",
    ]
    source = next((path for path in candidates if path.exists()), candidates[0])

    svg = extract_vector_svg_from_bytes(source.read_bytes())

    assert "<image " in svg
    assert "unsupported drawing primitives" not in svg
    assert "POSID_" not in svg


def test_gr_77775_svg_emits_multiple_class29_tile_images() -> None:
    from pathlib import Path  # noqa: PLC0415

    root = Path(__file__).resolve().parents[1]
    candidates = [
        root / "sample" / "GR-77775.cgm",
        root / "test_files" / "GR-77775.cgm",
    ]
    source = next((path for path in candidates if path.exists()), candidates[0])

    svg = extract_vector_svg_from_bytes(source.read_bytes())

    assert svg.count("<image ") >= 2
    assert 'preserveAspectRatio="none"' in svg


def test_gr_217420_id29_fallback_renders_image() -> None:
    from pathlib import Path  # noqa: PLC0415

    root = Path(__file__).resolve().parents[1]
    candidates = [
        root / "sample" / "GR-217420.cgm",
        root / "test_files" / "GR-217420.cgm",
    ]
    source = next((path for path in candidates if path.exists()), candidates[0])

    svg = extract_vector_svg_from_bytes(source.read_bytes())

    assert "<image " in svg


def test_gr_351121_id29_safe_raw_profile_renders_image() -> None:
    from pathlib import Path  # noqa: PLC0415

    root = Path(__file__).resolve().parents[1]
    candidates = [
        root / "sample" / "GR-351121.cgm",
        root / "test_files" / "GR-351121.cgm",
    ]
    source = next((path for path in candidates if path.exists()), candidates[0])

    svg = extract_vector_svg_from_bytes(source.read_bytes())

    assert "<image " in svg


def test_gr_77912_tiles_7_and_10_prefer_padded_fax4_width() -> None:
    from pathlib import Path  # noqa: PLC0415

    from PIL import Image, ImageChops  # noqa: PLC0415

    from cgm.extract.tiles import (  # noqa: PLC0415
        _cached_ccittfax4_decode,
        _decode_fax_output_to_bitmap,
        _decode_tile_payload_to_image,
        _parse_tile_arrays,
    )

    root = Path(__file__).resolve().parents[1]
    candidates = [
        root / "sample" / "GR-77912.cgm",
        root / "test_files" / "GR-77912.cgm",
    ]
    source = next((path for path in candidates if path.exists()), candidates[0])

    array = _parse_tile_arrays(source.read_bytes())[0]
    tile_width_obj = array.get("tile_width")
    tile_height_obj = array.get("tile_height")
    tiles_obj = array.get("tiles")
    assert isinstance(tile_width_obj, int)
    assert isinstance(tile_height_obj, int)
    assert isinstance(tiles_obj, list)
    tile_width = tile_width_obj
    tile_height = tile_height_obj

    def _decode_explicit_width(payload: bytes, actual_width: int) -> Image.Image:
        raw = _cached_ccittfax4_decode(payload, height=tile_height, width=actual_width)
        assert raw is not None
        if actual_width > tile_width:
            stripped = bytearray()
            for row_idx in range(tile_height):
                stripped.extend(raw[row_idx * actual_width : row_idx * actual_width + tile_width])
            raw = bytes(stripped)
        bits = _decode_fax_output_to_bitmap(raw, tile_width, tile_height)
        assert bits is not None
        pixels = bytes(0 if bit else 255 for bit in bits)
        return Image.frombytes("L", (tile_width, tile_height), pixels)

    for tile_index in (7, 10):
        tile = tiles_obj[tile_index]
        assert isinstance(tile, dict)
        payload = bytes(tile["payload"])
        current = _decode_tile_payload_to_image(
            payload,
            tile_width,
            tile_height,
            compression=tile["compression"],
            bit_order=tile["bit_order"],
            row_padding=tile["row_padding"],
            orientation=tile["orientation"],
        )
        nominal = _decode_explicit_width(payload, tile_width)
        padded = _decode_explicit_width(payload, tile_width + 8)

        assert current is not None
        assert ImageChops.difference(nominal, padded).getbbox() is not None
        assert ImageChops.difference(current, padded).getbbox() is None


def test_gr_74218_and_gr_76072_tiles_prefer_nominal_fax4_width() -> None:
    from pathlib import Path  # noqa: PLC0415

    from PIL import Image, ImageChops  # noqa: PLC0415

    from cgm.extract.tiles import (  # noqa: PLC0415
        _cached_ccittfax4_decode,
        _decode_fax_output_to_bitmap,
        _decode_tile_payload_to_image,
        _parse_tile_arrays,
    )

    root = Path(__file__).resolve().parents[1]
    cases = (
        ("GR-74218.cgm", 8),
        ("GR-76072.cgm", 4),
    )

    for file_name, tile_index in cases:
        candidates = [
            root / "sample" / file_name,
            root / "test_files" / file_name,
        ]
        source = next((path for path in candidates if path.exists()), candidates[0])
        array = _parse_tile_arrays(source.read_bytes())[0]
        tile_width_obj = array.get("tile_width")
        tile_height_obj = array.get("tile_height")
        tiles_obj = array.get("tiles")
        assert isinstance(tile_width_obj, int)
        assert isinstance(tile_height_obj, int)
        assert isinstance(tiles_obj, list)
        tile_width = tile_width_obj
        tile_height = tile_height_obj

        def _decode_explicit_width(
            payload: bytes,
            actual_width: int,
            image_width: int,
            image_height: int,
        ) -> Image.Image:
            raw = _cached_ccittfax4_decode(payload, height=image_height, width=actual_width)
            assert raw is not None
            if actual_width > image_width:
                stripped = bytearray()
                for row_idx in range(image_height):
                    stripped.extend(
                        raw[row_idx * actual_width : row_idx * actual_width + image_width]
                    )
                raw = bytes(stripped)
            bits = _decode_fax_output_to_bitmap(raw, image_width, image_height)
            assert bits is not None
            pixels = bytes(0 if bit else 255 for bit in bits)
            return Image.frombytes("L", (image_width, image_height), pixels)

        tile = tiles_obj[tile_index]
        assert isinstance(tile, dict)
        payload = bytes(tile["payload"])
        current = _decode_tile_payload_to_image(
            payload,
            tile_width,
            tile_height,
            compression=tile["compression"],
            bit_order=tile["bit_order"],
            row_padding=tile["row_padding"],
            orientation=tile["orientation"],
        )
        nominal = _decode_explicit_width(payload, tile_width, tile_width, tile_height)
        padded = _decode_explicit_width(payload, tile_width + 16, tile_width, tile_height)

        assert current is not None
        assert ImageChops.difference(nominal, padded).getbbox() is not None
        assert ImageChops.difference(current, nominal).getbbox() is None


def test_gr_78129_gr_78167_and_gr_78836_tiles_prefer_padded_fax4_width() -> None:
    from pathlib import Path  # noqa: PLC0415

    from PIL import Image, ImageChops  # noqa: PLC0415

    from cgm.extract.tiles import (  # noqa: PLC0415
        _cached_ccittfax4_decode,
        _decode_fax_output_to_bitmap,
        _decode_tile_payload_to_image,
        _parse_tile_arrays,
    )

    root = Path(__file__).resolve().parents[1]
    cases = (
        ("GR-78129.cgm", 3),
        ("GR-78129.cgm", 6),
        ("GR-78167.cgm", 3),
        ("GR-78167.cgm", 7),
        ("GR-78836.cgm", 6),
    )

    for file_name, tile_index in cases:
        candidates = [
            root / "sample" / file_name,
            root / "test_files" / file_name,
        ]
        source = next((path for path in candidates if path.exists()), candidates[0])
        array = _parse_tile_arrays(source.read_bytes())[0]
        tile_width_obj = array.get("tile_width")
        tile_height_obj = array.get("tile_height")
        tiles_obj = array.get("tiles")
        assert isinstance(tile_width_obj, int)
        assert isinstance(tile_height_obj, int)
        assert isinstance(tiles_obj, list)
        tile_width = tile_width_obj
        tile_height = tile_height_obj

        def _decode_explicit_width(
            payload: bytes,
            actual_width: int,
            image_width: int,
            image_height: int,
        ) -> Image.Image:
            raw = _cached_ccittfax4_decode(payload, height=image_height, width=actual_width)
            assert raw is not None
            if actual_width > image_width:
                stripped = bytearray()
                for row_idx in range(image_height):
                    stripped.extend(
                        raw[row_idx * actual_width : row_idx * actual_width + image_width]
                    )
                raw = bytes(stripped)
            bits = _decode_fax_output_to_bitmap(raw, image_width, image_height)
            assert bits is not None
            pixels = bytes(0 if bit else 255 for bit in bits)
            return Image.frombytes("L", (image_width, image_height), pixels)

        tile = tiles_obj[tile_index]
        assert isinstance(tile, dict)
        payload = bytes(tile["payload"])
        current = _decode_tile_payload_to_image(
            payload,
            tile_width,
            tile_height,
            compression=tile["compression"],
            bit_order=tile["bit_order"],
            row_padding=tile["row_padding"],
            orientation=tile["orientation"],
        )
        nominal = _decode_explicit_width(payload, tile_width, tile_width, tile_height)
        padded = _decode_explicit_width(payload, tile_width + 8, tile_width, tile_height)

        assert current is not None
        assert ImageChops.difference(nominal, padded).getbbox() is not None
        assert ImageChops.difference(current, padded).getbbox() is None


def test_gr_single_payload_id29_2010_uses_wrapper_tile_dimensions() -> None:
    from pathlib import Path  # noqa: PLC0415

    from cgm.extract.core import decode_vdc_extent  # noqa: PLC0415
    from cgm.extract.tiles import (  # noqa: PLC0415
        _decode_tile_payload_to_image,
        _parse_binary_tile_array_header,
    )

    root = Path(__file__).resolve().parents[1]
    cases = (
        "GR-295523.cgm",
        "GR-265523.cgm",
        "GR-240668.cgm",
        "GR-227821.cgm",
    )

    for file_name in cases:
        candidates = [
            root / "sample" / file_name,
            root / "test_files" / file_name,
        ]
        source = next((path for path in candidates if path.exists()), candidates[0])
        extent = None
        parsed_payload = None
        wrapper = None
        source_bytes = source.read_bytes()
        for element in iter_elements(source_bytes):
            if (element.class_id, element.element_id) == (2, 6):
                extent = decode_vdc_extent(element.parameters)
            if (element.class_id, element.element_id) == (0, 19):
                wrapper = _parse_binary_tile_array_header(element.parameters)
            if (element.class_id, element.element_id) == (4, 29):
                parsed_payload = _parse_element29_raster_prefix(element.parameters)
                if parsed_payload is not None:
                    break

        assert extent is not None
        assert wrapper is not None
        assert parsed_payload is not None
        _compression, row_padding, bit_order, orientation, payload = parsed_payload
        width = round(abs(extent[2] - extent[0]))
        height = round(abs(extent[3] - extent[1]))
        wrapper_width = int(wrapper["tile_width"])
        wrapper_height = int(wrapper["tile_height"])

        nominal = _decode_tile_payload_to_image(
            payload,
            width,
            height,
            compression=2,
            bit_order=bit_order,
            row_padding=row_padding,
            orientation=orientation,
        )
        rendered = _render_first_element29_raster(source_bytes)

        assert nominal is not None
        assert rendered is not None
        assert nominal.size == (width, height)
        assert rendered.size == (wrapper_width, wrapper_height)


def test_id29_prefix_families_have_no_new_frequent_format() -> None:
    from pathlib import Path  # noqa: PLC0415

    root = Path(__file__).resolve().parents[1] / "test_files"
    prefix_counts: Counter[tuple[int, int, int, int]] = Counter()

    for source in root.glob("*.cgm"):
        for element in iter_elements(source.read_bytes()):
            if element.class_id != 4 or element.element_id != 29:
                continue
            if decode_restricted_text(element.parameters) is not None:
                continue
            parsed = _parse_element29_raster_prefix(element.parameters)
            if parsed is None:
                continue
            prefix_counts[parsed[:4]] += 1

    known_families = {
        (2, 0, 1, 0),
        (6, 0, 8, 6),
        (7, 0, 8, 6),
    }
    unknown_families = {
        prefix: count for prefix, count in prefix_counts.items() if prefix not in known_families
    }

    if unknown_families:
        assert max(unknown_families.values()) <= 2


def test_0800c8af84b96799_gdp_alignment_avoids_collapsed_viewbox() -> None:
    from pathlib import Path  # noqa: PLC0415

    root = Path(__file__).resolve().parents[1]
    source = root / "test_files" / "0800c8af84b96799.cgm"

    svg = extract_vector_svg_from_bytes(source.read_bytes())

    assert 'viewBox="-0.245 -0.000 1.000 1.000"' not in svg


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
    match = re.search(r'viewBox="([^"]+)"', svg)
    assert match is not None
    _x, _y, width, height = (float(part) for part in match.group(1).split())
    assert width < 10_000_000
    assert height < 1_000_000


def test_gr_78946_does_not_emit_noisy_massive_polylines() -> None:
    from pathlib import Path  # noqa: PLC0415

    root = Path(__file__).resolve().parents[1]
    candidates = [
        root / "sample" / "GR-78946.cgm",
        root / "test_files" / "GR-78946.cgm",
    ]
    source = next((path for path in candidates if path.exists()), candidates[0])

    svg = extract_vector_svg_from_bytes(source.read_bytes())

    point_cloud_sizes = [
        len(points.split()) for points in re.findall(r'<polyline points="([^"]+)"', svg)
    ]
    if point_cloud_sizes:
        assert max(point_cloud_sizes) < 200
    else:
        assert "<polygon" in svg


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
    assert images[0].local_color_precision == 16
    assert images[0].cell_representation_mode == 0
    assert images[0].payload == bytes.fromhex(hex_payload)


def test_extract_raw_images_from_clear_text_color_tiles() -> None:
    hex_payload = "00112233445566778899AABBCCDDEEFF"
    data = (
        "BEGTILEARRAY 0 0 0 0 1 1 2 2 0 0 0 0 2 2; "
        f"COLORTILE 2 16 0 1 '' {hex_payload}; "
        "ENDTILEARRAY;"
    ).encode("ascii")

    images = extract_raw_images_from_bytes(data)

    assert len(images) == 1
    assert images[0].width == 2
    assert images[0].height == 2
    assert images[0].local_color_precision == 16
    assert images[0].cell_representation_mode == 0
    assert images[0].payload == bytes.fromhex(hex_payload)


def test_extract_raw_images_from_clear_text_direct_color_tiles_default_precision() -> None:
    hex_payload = "00112233445566778899AABBCCDDEEFF"
    data = (
        f"BEGTILEARRAY 0 0 0 0 1 1 2 2 0 0 0 0 2 2; DIRECTCOLORTILE '' {hex_payload}; ENDTILEARRAY;"
    ).encode("ascii")

    images = extract_raw_images_from_bytes(data)

    assert len(images) == 1
    assert images[0].width == 2
    assert images[0].height == 2
    assert images[0].local_color_precision == 24
    assert images[0].cell_representation_mode == 0
    assert images[0].payload == bytes.fromhex(hex_payload)


def test_extract_vector_svg_from_bytes_renders_rgb_cell_array() -> None:
    # 2x1 RGB payload: red pixel then green pixel.
    rgb_payload = bytes([255, 0, 0, 0, 255, 0])
    params = (
        (b"\x00" * 12)
        + (2).to_bytes(2, "big")
        + (1).to_bytes(2, "big")
        + (24).to_bytes(2, "big")
        + (0).to_bytes(2, "big")
        + rgb_payload
    )
    data = _header(4, 9, len(params)) + params

    svg = extract_vector_svg_from_bytes(data)

    assert "<svg" in svg
    assert "<image" in svg


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


def test_extract_rendered_images_to_directory_stitches_color_tiles(tmp_path: Path) -> None:
    src = tmp_path / "color_tiles.cgm"
    # 2x2 RGBA tile (16 bytes => 32 hex chars).
    hex_payload = "FF0000FF00FF00FF0000FFFFFFFFFFFF"
    src.write_text(
        "BEGTILEARRAY 0 0 0 0 1 1 2 2 0 0 0 0 2 2; "
        f"COLORTILE 2 32 0 1 '' {hex_payload}; "
        "ENDTILEARRAY;",
        encoding="ascii",
    )

    written = extract_rendered_images_to_directory(src, tmp_path, stem="color_tiles")

    svg_path = tmp_path / "color_tiles_0000.svg"
    assert svg_path in written
    svg = svg_path.read_text(encoding="utf-8")
    assert "<svg" in svg
    assert "<image" in svg


def test_extract_vector_svg_from_bytes_supports_clear_text_color_tile_arrays() -> None:
    tile_0 = "FF0000FF" * 8
    tile_1 = "00FF00FF" * 8
    data = (
        "BEGTILEARRAY 0 0 0 0 2 1 4 2 0 0 0 0 8 2; "
        f"COLORTILE 2 8 0 1 '' {tile_0}; "
        f"COLORTILE 2 8 0 1 '' {tile_1}; "
        "ENDTILEARRAY;"
    ).encode("ascii")

    svg = extract_vector_svg_from_bytes(data)

    assert svg.count("<image") == 2
    assert 'viewBox="0.000 0.000 8.000 2.000"' in svg


def test_decode_tile_payload_to_image_supports_indexed_tiles() -> None:
    image = extract_module._decode_tile_payload_to_image(
        bytes([0, 1, 2, 3]),
        2,
        2,
        family="indexed",
    )

    assert image is not None
    assert image.mode == "P"
    assert image.size == (2, 2)


def test_render_raw_image_payload_uses_supplied_palette_colors() -> None:
    raw = extract_module.RawImage(
        index=0,
        element_offset=0,
        payload=bytes([0, 1]),
        width=2,
        height=1,
        local_color_precision=8,
        cell_representation_mode=0,
    )
    palette = bytearray(value for value in range(256) for _ in range(3))
    palette[3:6] = bytes([255, 0, 0])

    image = extract_module._render_raw_image_payload(raw, indexed_palette=bytes(palette))

    assert image is not None
    rgb = image.convert("RGB")
    assert rgb.getpixel((0, 0)) == (0, 0, 0)
    assert rgb.getpixel((1, 0)) == (255, 0, 0)


def test_render_first_tile_array_uses_color_table_palette() -> None:
    tile_payload = bytes([0, 1, 0, 1])
    color_table = {1: (255, 0, 0)}
    palette = extract_module._indexed_palette_bytes(color_table)

    image = extract_module._decode_tile_payload_to_image(
        tile_payload,
        2,
        2,
        family="indexed",
        indexed_palette=palette,
    )

    assert image is not None
    rgb = image.convert("RGB")
    assert rgb.getpixel((0, 0)) == (0, 0, 0)
    assert rgb.getpixel((1, 0)) == (255, 0, 0)


def test_decode_tile_payload_to_image_scales_16bit_direct_color_extent() -> None:
    # One red pixel in 12-bit domain carried in 16-bit channels (max=4095).
    payload = bytes.fromhex("0fff00000000")
    image = extract_module._decode_tile_payload_to_image(
        payload,
        1,
        1,
        local_color_precision=16,
        color_value_extent=(0, 0, 0, 4095, 4095, 4095),
    )

    assert image is not None
    assert image.mode == "RGB"
    assert image.getpixel((0, 0)) == (255, 0, 0)


def test_render_raw_image_payload_scales_16bit_direct_color_extent() -> None:
    raw = extract_module.RawImage(
        index=0,
        element_offset=0,
        payload=bytes.fromhex("0fff0000000000000fff0000"),
        width=2,
        height=1,
        local_color_precision=16,
        cell_representation_mode=0,
    )

    image = extract_module._render_raw_image_payload(
        raw,
        color_value_extent=(0, 0, 0, 4095, 4095, 4095),
    )

    assert image is not None
    assert image.mode == "RGB"
    assert image.getpixel((0, 0)) == (255, 0, 0)
    assert image.getpixel((1, 0)) == (0, 255, 0)


def test_decode_tile_payload_to_image_scales_nonzero_16bit_extent_and_clamps() -> None:
    # Extent is 1024..3072 for each channel.
    # Pixel 0: (0,1024,3072) -> (0,0,255) after clamp/scale.
    # Pixel 1: (3072,2048,65535) -> (255,128,255) after clamp/scale.
    payload = bytes.fromhex("000004000c000c000800ffff")

    image = extract_module._decode_tile_payload_to_image(
        payload,
        2,
        1,
        local_color_precision=16,
        color_value_extent=(1024, 1024, 1024, 3072, 3072, 3072),
    )

    assert image is not None
    assert image.mode == "RGB"
    assert image.getpixel((0, 0)) == (0, 0, 255)
    assert image.getpixel((1, 0)) == (255, 128, 255)


def test_render_raw_image_payload_scales_nonzero_16bit_extent_and_clamps() -> None:
    raw = extract_module.RawImage(
        index=0,
        element_offset=0,
        payload=bytes.fromhex("000004000c000c000800ffff"),
        width=2,
        height=1,
        local_color_precision=16,
        cell_representation_mode=0,
    )

    image = extract_module._render_raw_image_payload(
        raw,
        color_value_extent=(1024, 1024, 1024, 3072, 3072, 3072),
    )

    assert image is not None
    assert image.mode == "RGB"
    assert image.getpixel((0, 0)) == (0, 0, 255)
    assert image.getpixel((1, 0)) == (255, 128, 255)


def test_extract_vector_svg_from_bytes_supports_clear_text_index_tile_arrays() -> None:
    payload = "000102030405060708090A0B0C0D0E0F"
    data = (
        "BEGTILEARRAY 0 0 0 0 1 1 2 2 0 0 0 0 2 2; "
        f"INDEXCOLORTILE 2 8 0 1 '' {payload}; "
        "ENDTILEARRAY;"
    ).encode("ascii")

    svg = extract_vector_svg_from_bytes(data)

    assert "<image" in svg
    assert 'viewBox="0.000 0.000 2.000 2.000"' in svg


def test_extract_vector_svg_from_bytes_supports_short_clear_text_index_tile_payload() -> None:
    data = (
        "BEGTILEARRAY 0 0 0 0 1 1 2 1 0 0 0 0 2 1; INDEXCOLORTILE 2 8 0 1 '' 0001; ENDTILEARRAY;"
    ).encode("ascii")

    svg = extract_vector_svg_from_bytes(data)

    assert "<image" in svg
    assert 'viewBox="0.000 0.000 2.000 1.000"' in svg


def test_parse_text_tile_arrays_gr_77775_payloads_are_not_truncated() -> None:
    from pathlib import Path  # noqa: PLC0415

    root = Path(__file__).resolve().parents[1]
    candidates = [
        root / "sample" / "GR-77775.cgm",
        root / "test_files" / "GR-77775.cgm",
    ]
    source = next((path for path in candidates if path.exists()), candidates[0])

    arrays = extract_module._parse_text_tile_arrays(source.read_bytes())

    assert len(arrays) == 1
    tiles = arrays[0].get("tiles")
    assert isinstance(tiles, list)
    assert len(tiles) == 12
    lengths = [len(tile.get("payload", b"")) for tile in tiles if isinstance(tile, dict)]
    assert len(lengths) == 12
    # Regression guard: truncated parser output previously produced tiny payloads (e.g. 19 bytes).
    assert min(lengths) >= 200


def test_extract_vector_svg_from_clear_text_scales_direct_color_tile_with_color_value_extent() -> (
    None
):
    # Two pixels in 16-bit RGB channels with extent 1024..3072 for each channel:
    # (0,1024,3072) -> (0,0,255), (3072,2048,65535) -> (255,128,255)
    hex_payload = "000004000c000c000800ffff"
    data = (
        "COLRVALUEEXT 1024 1024 1024 3072 3072 3072; "
        "BEGTILEARRAY 0 0 0 0 1 1 2 1 0 0 0 0 2 1; "
        f"DIRECTCOLORTILE 2 16 0 1 '' {hex_payload}; "
        "ENDTILEARRAY;"
    ).encode("ascii")

    svg = extract_vector_svg_from_bytes(data)

    match = re.search(r'href="(data:image/png;base64,[^"]+)"', svg)
    assert match is not None
    href = match.group(1)
    assert href.startswith("data:image/png;base64,")
    encoded = href.split(",", 1)[1]
    decoded = base64.b64decode(encoded)
    image_module = importlib.import_module("PIL.Image")
    image = image_module.open(io.BytesIO(decoded)).convert("RGB")

    assert image.size == (2, 1)
    assert image.getpixel((0, 0)) == (0, 0, 255)
    assert image.getpixel((1, 0)) == (255, 128, 255)


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
