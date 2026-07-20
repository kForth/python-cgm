# python-cgm

[![GitHub](https://img.shields.io/badge/github-repo-blue?logo=github)](https://github.com/kForth/python-cgm)
[![GitHub License](https://img.shields.io/github/license/kforth/python-cgm)](https://github.com/kForth/python-cgm/blob/main/LICENSE)
[![GitHub Forks](https://img.shields.io/github/forks/kforth/python-cgm)](https://github.com/kForth/python-cgm/forks)
[![GitHub Stars](https://img.shields.io/github/stars/kforth/python-cgm)](https://github.com/kForth/python-cgm/stargazers)

[![PyPI Version](https://img.shields.io/pypi/v/python-cgm?logo=python&logoColor=white)](https://pypi.org/p/python-cgm)
![Pepy Total Downloads](https://img.shields.io/pepy/dt/python-cgm)
![PyPI Downloads](https://img.shields.io/pypi/dm/python-cgm)


Read-only Python tools for parsing binary and clear-text CGM (ISO/IEC 8632) files,
producing final SVG output with optional raster tile backgrounds, and extracting
hotspot metadata as JSON.

This package focuses on practical CGM extraction workflows: parsing CGM content,
extracting image-bearing `Cell Array` payloads, decoding clear-text tile arrays,
composing raster+vector SVG output, and recovering hotspots from APD region
properties and APS geometry fallback.
It does not support writing CGM files.

## Installation

Install the latest version using `pip`:

```bash
pip install python-cgm
```

## What It Does

- Parses binary and clear-text CGM command streams.
- Finds `Cell Array` elements (class 4, element 9) and extracts their raw payload bytes.
- Decodes clear-text tiled bitonal, indexed, and direct-color arrays.
- Builds a final SVG output that can include an embedded raster background.
- Converts vector-like CGM drawing primitives into SVG overlays.
- Extracts hotspots from APD `name`/`region` records and APS geometry groups.
- Exports parsed element data, payload metadata, rendered SVG, and hotspots as JSON.

## Quick Start

```python
from cgm import (
    extract_data_json,
    extract_final_image_and_hotspots,
    extract_hotspots,
    extract_raw_images,
    extract_raw_images_to_directory,
    extract_vector_svg,
)

images = extract_raw_images("drawing.cgm")
print(f"Found {len(images)} raster payload(s)")

for image in images:
    print(
        image.index,
        image.element_offset,
        image.width,
        image.height,
        len(image.payload),
    )

written = extract_raw_images_to_directory("drawing.cgm", "./out")
print("Wrote", len(written), "payload file(s)")

svg = extract_vector_svg("drawing.cgm")
print("SVG length:", len(svg))

snapshot_json = extract_data_json("drawing.cgm")
print("JSON length:", len(snapshot_json))

final = extract_final_image_and_hotspots("drawing.cgm")
print("Final SVG length:", len(final["image"]))
print("Hotspots:", len(final["hotspots"]))

hotspots = extract_hotspots("drawing.cgm")
print("Hotspot objects:", len(hotspots))
```

## CLI

After installation, use the CLI to export the final SVG and hotspot JSON:

```bash
cgm-extract file.cgm ./out
```

By default this writes:

- `<basename>_0000.svg`
- `<basename>_0000.hotspots.json`

With debug enabled it also writes:

- `<basename>_decode_report.json`

Optional flag:

```bash
cgm-extract file.cgm ./out --debug
```

## API

- `extract_raw_images(file_path) -> list[RawImage]`
- `extract_raw_images_from_bytes(data) -> list[RawImage]`
- `extract_raw_images_to_directory(file_path, output_dir, stem="image") -> list[Path]`
- `extract_rendered_images_to_directory(file_path, output_dir, stem="image", debug_report=False) -> list[Path]`
- `extract_vector_svg(file_path) -> str`
- `extract_vector_svg_from_bytes(data) -> str`
- `extract_vector_svg_to_directory(file_path, output_dir, stem="image") -> Path`
- `extract_data_json(file_path) -> str`
- `extract_data_json_from_bytes(data) -> str`
- `extract_data_json_to_directory(file_path, output_dir, stem="image") -> Path`
- `extract_hotspots(file_path) -> list[HotSpot]`
- `extract_hotspots_from_bytes(data) -> list[HotSpot]`
- `extract_hotspots_to_directory(file_path, output_dir, stem="image") -> Path`
- `extract_final_image_and_hotspots(file_path) -> dict[str, object]`

`RawImage` fields:

- `index`: zero-based image index.
- `element_offset`: byte offset of the CGM element in the source file.
- `payload`: raw image payload bytes.
- `width` / `height`: dimensions when present in common binary or clear-text tile layouts.
- `local_color_precision`: declared color precision for the payload when available.
- `cell_representation_mode`: declared cell representation mode when available.

## Scope And Limitations

- Supports binary and clear-text CGM streams used by this project.
- Raster decoding/composition depends on the declared CGM payload encoding and tile metadata.
- SVG output covers the supported CGM command patterns in the file.
- Hotspot extraction supports APD region records plus APS geometry fallback.
- JSON output can be large because it includes full element parameter/payload hex data.
- Does **not** support CGM writing.

## License

python-cgm (C) 2026 Kestin Goforth.

This project is licensed under the BSD 3-Clause License - see the [license file](LICENSE) for details.
