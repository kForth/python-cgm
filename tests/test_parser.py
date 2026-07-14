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
    assert elements[0].parameters[20:] == bytes.fromhex(hex_payload)
