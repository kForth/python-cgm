"""Raw Cell Array raster extraction helpers."""

from __future__ import annotations

import logging
from pathlib import Path

from .common import _parse_cell_array_hints
from .parser import CELL_ARRAY_CLASS_ID, CELL_ARRAY_ELEMENT_ID, iter_elements
from .types import RawImage

log = logging.getLogger("cgm.raw_images")


def extract_raw_images_from_bytes(data: bytes) -> list[RawImage]:
    """Extract raw raster payloads from Cell Array elements in a CGM stream."""
    images: list[RawImage] = []
    image_index = 0
    element_histogram: dict[tuple[int, int], int] = {}

    if log.isEnabledFor(logging.DEBUG):
        log.debug("Starting extraction from %d bytes", len(data))

    for element in iter_elements(data):
        element_key = (element.class_id, element.element_id)
        element_histogram[element_key] = element_histogram.get(element_key, 0) + 1

        if element.class_id != CELL_ARRAY_CLASS_ID or element.element_id != CELL_ARRAY_ELEMENT_ID:
            continue

        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                "Found Cell Array element: offset=%d param_length=%d",
                element.offset,
                len(element.parameters),
            )

        width, height, payload_offset = _parse_cell_array_hints(element.parameters)
        payload = (
            element.parameters[payload_offset:] if payload_offset < len(element.parameters) else b""
        )

        if log.isEnabledFor(logging.DEBUG):
            log.debug(
                (
                    "Extracted image payload: index=%d offset=%d width=%s height=%s "
                    "payload_offset=%d payload_size=%d"
                ),
                image_index,
                element.offset,
                width,
                height,
                payload_offset,
                len(payload),
            )

        images.append(
            RawImage(
                index=image_index,
                element_offset=element.offset,
                payload=payload,
                width=width,
                height=height,
            )
        )
        image_index += 1

    if log.isEnabledFor(logging.DEBUG):
        if images:
            log.debug("Extraction complete: found %d Cell Array image(s)", len(images))
        else:
            top_ids = ", ".join(
                f"({class_id},{element_id})={count}"
                for (class_id, element_id), count in sorted(
                    element_histogram.items(), key=lambda item: item[1], reverse=True
                )[:10]
            )
            log.debug(
                "No Cell Array elements (class=%d id=%d) found in %d parsed elements. "
                "Most frequent element IDs: %s",
                CELL_ARRAY_CLASS_ID,
                CELL_ARRAY_ELEMENT_ID,
                sum(element_histogram.values()),
                top_ids,
            )

    return images


def extract_raw_images(file_path: str | Path) -> list[RawImage]:
    path = Path(file_path)
    raw = path.read_bytes()
    if log.isEnabledFor(logging.DEBUG):
        log.debug("Loaded CGM file %s (%d bytes)", path, len(raw))
    return extract_raw_images_from_bytes(raw)


def extract_raw_images_to_directory(
    file_path: str | Path,
    output_dir: str | Path,
    *,
    stem: str = "image",
) -> list[Path]:
    written: list[Path] = []
    if log.isEnabledFor(logging.DEBUG):
        log.debug("Writing extracted payloads to %s with stem='%s'", output_dir, stem)

    for image in extract_raw_images(file_path):
        path = image.write(output_dir, stem=stem)
        if log.isEnabledFor(logging.DEBUG):
            log.debug("Wrote payload file: %s (%d bytes)", path, len(image.payload))
        written.append(path)

    if log.isEnabledFor(logging.DEBUG):
        log.debug("Finished writing payload files: %d file(s)", len(written))

    return written
