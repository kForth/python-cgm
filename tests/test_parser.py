from __future__ import annotations

__author__ = "Kestin Goforth"
__copyright__ = "Copyright 2026"
__license__ = "BSD-3-Clause"

import pytest

from cgm.errors import CGMParseError
from cgm.parser import iter_elements


def _header(class_id: int, element_id: int, length: int) -> bytes:
    value = (class_id << 12) | (element_id << 5) | length
    return value.to_bytes(2, "big")


def test_iter_elements_parses_short_form_with_padding() -> None:
    params = b"abc"
    data = _header(4, 9, len(params)) + params + b"\x00"

    elements = list(iter_elements(data))

    assert len(elements) == 1
    element = elements[0]
    assert element.class_id == 4
    assert element.element_id == 9
    assert element.parameters == params
    assert element.offset == 0


def test_iter_elements_returns_no_elements_for_empty_input() -> None:
    elements = list(iter_elements(b""))

    assert elements == []


def test_iter_elements_treats_binary_data_with_semicolons_as_binary() -> None:
    params = b"abc;def"
    data = _header(4, 9, len(params)) + params + b"\x00"

    elements = list(iter_elements(data))

    assert len(elements) == 1
    assert elements[0].class_id == 4
    assert elements[0].element_id == 9
    assert elements[0].parameters == params


def test_iter_elements_parses_long_form_chunked_parameters() -> None:
    payload = b"hello"
    chunk_header = len(payload).to_bytes(2, "big")
    data = _header(2, 3, 0x1F) + chunk_header + payload + b"\x00"

    elements = list(iter_elements(data))

    assert len(elements) == 1
    assert elements[0].class_id == 2
    assert elements[0].element_id == 3
    assert elements[0].parameters == payload


def test_iter_elements_raises_on_truncated_parameter_block() -> None:
    data = _header(4, 9, 4) + b"xy"

    with pytest.raises(CGMParseError):
        list(iter_elements(data))


def test_iter_elements_parses_clear_text_line() -> None:
    data = b'LINE 0 0 10 10;\nTEXT 5 6 "HELLO";\n'

    elements = list(iter_elements(data))

    assert len(elements) == 2
    assert elements[0].class_id == 4
    assert elements[0].element_id == 1
    assert len(elements[0].parameters) == 16
    assert elements[1].class_id == 4
    assert elements[1].element_id == 5


def test_iter_elements_parses_clear_text_bitonal_tile_as_cell_array() -> None:
    hex_payload = "0123456789ABCDEFFEDCBA9876543210"
    data = (
        "BEGTILEARRAY 0 0 0 0 1 1 2 2 0 0 0 0 2 2; "
        f"BITONALTILE 2 16 0 1 '' {hex_payload}; "
        "ENDTILEARRAY;"
    ).encode("ascii")

    elements = list(iter_elements(data))

    assert len(elements) == 1
    assert elements[0].class_id == 4
    assert elements[0].element_id == 9
    assert int.from_bytes(elements[0].parameters[16:18], "big") == 16
    assert int.from_bytes(elements[0].parameters[18:20], "big") == 0
    assert elements[0].parameters[20:] == bytes.fromhex(hex_payload)


def test_iter_elements_parses_clear_text_color_tile_as_cell_array() -> None:
    hex_payload = "00112233445566778899AABBCCDDEEFF"
    data = (
        "BEGTILEARRAY 0 0 0 0 1 1 2 2 0 0 0 0 2 2; "
        f"COLORTILE 2 16 0 1 '' {hex_payload}; "
        "ENDTILEARRAY;"
    ).encode("ascii")

    elements = list(iter_elements(data))

    assert len(elements) == 1
    assert elements[0].class_id == 4
    assert elements[0].element_id == 9
    assert int.from_bytes(elements[0].parameters[16:18], "big") == 16
    assert int.from_bytes(elements[0].parameters[18:20], "big") == 0
    assert elements[0].parameters[20:] == bytes.fromhex(hex_payload)


def test_iter_elements_parses_clear_text_short_index_tile_payload() -> None:
    data = (
        "BEGTILEARRAY 0 0 0 0 1 1 2 1 0 0 0 0 2 1; INDEXCOLORTILE 2 8 0 1 '' 0001; ENDTILEARRAY;"
    ).encode("ascii")

    elements = list(iter_elements(data))

    assert len(elements) == 1
    assert elements[0].class_id == 4
    assert elements[0].element_id == 9
    assert int.from_bytes(elements[0].parameters[12:14], "big") == 2
    assert int.from_bytes(elements[0].parameters[14:16], "big") == 1
    assert int.from_bytes(elements[0].parameters[16:18], "big") == 8
    assert elements[0].parameters[20:] == bytes([0, 1])


def test_iter_elements_parses_direct_color_tile_with_default_precision() -> None:
    hex_payload = "00112233445566778899AABBCCDDEEFF"
    data = (
        f"BEGTILEARRAY 0 0 0 0 1 1 2 2 0 0 0 0 2 2; DIRECTCOLORTILE '' {hex_payload}; ENDTILEARRAY;"
    ).encode("ascii")

    elements = list(iter_elements(data))

    assert len(elements) == 1
    assert elements[0].class_id == 4
    assert elements[0].element_id == 9
    assert int.from_bytes(elements[0].parameters[16:18], "big") == 24
    assert int.from_bytes(elements[0].parameters[18:20], "big") == 0
    assert elements[0].parameters[20:] == bytes.fromhex(hex_payload)


def test_iter_elements_parses_clear_text_drawing_primitives() -> None:
    data = (
        "DISJOINTPOLYLINE 0 0 10 10; "
        "POLYMARKER 5 5 7 7; "
        "POLYGONSET 0 0 10 0 10 10 0 10; "
        "RECTANGLE 1 2 3 4; "
        "CIRCLE 20 30 5; "
        "ARC3PT 0 0 5 10 10 0; "
        "ARCCENTRE 50 50 55 50 50 55; "
        "ELLIPSE 70 70 80 70 70 90; "
        "ELLIPARC 90 90 100 90 90 100 95 90 90 95; "
        "GDP 1 2 3 4;"
    ).encode("ascii")

    elements = list(iter_elements(data))

    assert [element.class_id for element in elements] == [4] * 10
    assert [element.element_id for element in elements] == [2, 3, 8, 11, 12, 13, 15, 17, 18, 10]


def test_iter_elements_parses_clear_text_transparency_commands() -> None:
    data = b"TRANSPARENCY OFF; TRANSPARENCYMODE ON; TRANSPMODE 0;"

    elements = list(iter_elements(data))

    assert len(elements) == 3
    assert [element.class_id for element in elements] == [3, 3, 3]
    assert [element.element_id for element in elements] == [4, 4, 4]
    assert [element.parameters for element in elements] == [b"\x00", b"\x01", b"\x00"]


def test_iter_elements_parses_clear_text_clipping_commands() -> None:
    data = b"CLIPRECT 1 2 11 22; CLIPIND ON; CLIPIND OFF;"

    elements = list(iter_elements(data))

    assert len(elements) == 3
    assert [element.class_id for element in elements] == [3, 3, 3]
    assert [element.element_id for element in elements] == [5, 6, 6]
    assert elements[0].parameters == b"\x00\x01\x00\x02\x00\x0b\x00\x16"
    assert elements[1].parameters == b"\x01"
    assert elements[2].parameters == b"\x00"


def test_iter_elements_parses_clear_text_color_value_extent() -> None:
    data = b"COLRVALUEEXT 0 0 0 4095 4095 4095;"

    elements = list(iter_elements(data))

    assert len(elements) == 1
    assert elements[0].class_id == 1
    assert elements[0].element_id == 10
    assert elements[0].parameters == b"\x00\x00\x00\x00\x00\x00\x0f\xff\x0f\xff\x0f\xff"
