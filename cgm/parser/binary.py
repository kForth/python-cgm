"""Binary CGM element decoding."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from cgm.errors import CGMParseError
from cgm.parser.constants import LONG_FORM_SENTINEL
from cgm.types import CGMElement

if TYPE_CHECKING:
    from collections.abc import Iterator

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


def iter_binary_elements(data: bytes) -> Iterator[CGMElement]:
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
