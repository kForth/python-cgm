# python-cgm

[![GitHub](https://img.shields.io/badge/github-repo-blue?logo=github)](https://github.com/kForth/python-cgm)
[![GitHub License](https://img.shields.io/github/license/kforth/python-cgm)](https://github.com/kForth/python-cgm/blob/main/LICENSE)
[![GitHub Forks](https://img.shields.io/github/forks/kforth/python-cgm)](https://github.com/kForth/python-cgm/forks)
[![GitHub Stars](https://img.shields.io/github/stars/kforth/python-cgm)](https://github.com/kForth/python-cgm/stargazers)

[![PyPI Version](https://img.shields.io/pypi/v/python-cgm?logo=python&logoColor=white)](https://pypi.org/p/python-cgm)
![Pepy Total Downloads](https://img.shields.io/pepy/dt/python-cgm)
![PyPI Downloads](https://img.shields.io/pypi/dm/python-cgm)


Read-only Python tools for parsing binary CGM (ISO/IEC 8632) files and exporting
image-related data as raw payloads, SVG, and JSON.

This package focuses on **binary CGM parsing** and extraction of image-bearing
`Cell Array` elements, plus best-effort vector conversion for vector-oriented
CGM content. It does not support writing CGM files.

## Installation

Install the latest version using `pip`:

```bash
pip install python-cgm
```

## What It Does

- Parses binary CGM command streams.
- Finds `Cell Array` elements (class 4, element 9).
- Extracts raw payload bytes from each image-bearing element.
- Converts vector-like CGM drawing primitives into a best-effort SVG.
- Exports parsed element data, payload metadata, and embedded SVG as JSON.

## Quick Start

```python
from cgm import (
    extract_data_json,
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
```

## CLI

After installation, use the CLI to extract payloads directly:

```bash
cgm-extract file.cgm ./out
```

Export vector SVG:

```bash
cgm-extract file.cgm ./out --svg
```

Export parsed data snapshot as JSON:

```bash
cgm-extract file.cgm ./out --json
```

Export both SVG and JSON in one run:

```bash
cgm-extract file.cgm ./out --svg --json
```

Optional flags:

```bash
cgm-extract file.cgm ./out --stem payload
cgm-extract file.cgm ./out --debug
```

## API

- `extract_raw_images(file_path) -> list[RawImage]`
- `extract_raw_images_from_bytes(data) -> list[RawImage]`
- `extract_raw_images_to_directory(file_path, output_dir, stem="image") -> list[Path]`
- `extract_vector_svg(file_path) -> str`
- `extract_vector_svg_from_bytes(data) -> str`
- `extract_vector_svg_to_directory(file_path, output_dir, stem="image") -> Path`
- `extract_data_json(file_path) -> str`
- `extract_data_json_from_bytes(data) -> str`
- `extract_data_json_to_directory(file_path, output_dir, stem="image") -> Path`

`RawImage` fields:

- `index`: zero-based image index.
- `element_offset`: byte offset of the CGM element in the source file.
- `payload`: raw image payload bytes.
- `width` / `height`: best-effort dimensions when present in common binary layouts.

## Scope And Limitations

- Supports **binary CGM** streams.
- Provides **raw payload extraction** for raster data, not full pixel decoding to PNG/JPEG.
- SVG output is **best effort** and depends on CGM command patterns in the file.
- JSON output can be large because it includes full element parameter/payload hex data.
- Does **not** support CGM writing.

## License

python-cgm (C) 2026 Kestin Goforth.

This project is licensed under the BSD 3-Clause License - see the [license file](LICENSES/BSD-3-Clause.txt) for details.
