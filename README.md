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

## Supported CGM Elements And Features

The module focuses on practical extraction/rendering coverage for common binary
and clear-text CGM workflows.

### Binary CGM Element Coverage

- `class 1, id 3` (`VDC Type`): used to choose coordinate decoding path.
- `class 1, id 10` (`Color Value Extent`): used for 16-bit direct-color scaling.
- `class 1, id 11` (`VDC Integer Precision`): used for strict integer VDC decode.
- `class 1, id 12` (`VDC Real Precision`): used for strict real VDC decode.
- `class 2, id 6` (`VDC Extent`): used to set SVG view extents and raster placement.
- `class 3, id 4` (`Transparency`): mapped to SVG background behavior.
- `class 3, id 5` (`Clip Rectangle`): mapped to SVG clip paths.
- `class 3, id 6` (`Clip Indicator`): enables/disables clipping.
- `class 4, id 1` (`Polyline`): rendered to SVG polylines.
- `class 4, id 2` (`Disjoint Polyline`): rendered as segment polylines.
- `class 4, id 3` (`Polymarker`): rendered as SVG marker circles.
- `class 4, id 4` (`Text` continuation context): appended to prior text where applicable.
- `class 4, id 5` (`Text`): rendered to SVG text.
- `class 4, id 6` (`Append Text`): appended to prior text runs.
- `class 4, id 7` (`Polygon`): rendered to SVG polygons.
- `class 4, id 8` (`Polygon Set`): rendered as polygon geometry.
- `class 4, id 9` (`Cell Array`): extracted as `RawImage` payloads and used as raster candidates.
- `class 4, id 10` and `id 26` (`GDP`-like primitives): decoded as polyline-style vectors.
- `class 4, id 11` (`Rectangle`): rendered as SVG rect.
- `class 4, id 12` (`Circle`): rendered as SVG circle.
- `class 4, ids 13-16, 18-25, 27` (arc families): rendered as best-effort polyline geometry.
- `class 4, id 17` (`Ellipse`): rendered as SVG ellipse.
- `class 4, id 28`: parsed but no strict vector fallback rendering.
- `class 4, id 29` (`Restricted Text`): rendered when text payload decodes.
- `class 5, id 3` (`Line Width`): applied to SVG stroke width.
- `class 5, id 4` (`Line Color`): applied via palette/index mapping.
- `class 5, id 15` (`Character Height`): applied to SVG text size.
- `class 5, id 34` (`Color Table`): used for indexed palette and color mapping.
- `class 9, id 1` (`Application Data` / APD): used for hotspot `name`/`region` extraction.
- `class 0, id 21/22/23` (APS begin/end forms): used for hotspot grouping.

### Clear-Text Command Coverage

- Vector primitives: `LINE`, `POLYLINE`, `DISJOINTPOLYLINE`, `POLYMARKER`,
  `POLYGON`, `POLYGONSET`, `RECTANGLE`, `CIRCLE`, `ARC3PT`, `ARCCENTRE`,
  `ELLIPSE`, `ELLIPARC`, `GDP`.
- Text primitives: `TEXT`, `APPENDTEXT`, `RESTRICTEDTEXT`.
- Raster/tile commands: `CELLARRAY`, `BEGTILEARRAY`/`ENDTILEARRAY`,
  `BITONALTILE`, `MONOCHROMETILE`, `INDEXCOLORTILE`, `COLORTILE`,
  `DIRECTCOLORTILE` (and colour spelling variants).
- Attributes/control: `VDCEXT`, `COLRVALUEEXT`, `COLRTABLE`, `LINECOLR`,
  `TRANSPARENCY`, `CLIPRECT`, `CLIPIND`.
- Hotspot-related data: `BEGAPS`, `APD`, `ENDAPS`.

### Raster Decoding Features

- Extracts raw `Cell Array` payload bytes with metadata where present.
- Decodes bitonal raster data for uncompressed, CCITT Group 3, and CCITT Group 4 paths.
- Decodes indexed-color and direct-color tile payloads when dimensions/precision are usable.
- Composes raster backgrounds into SVG (embedded PNG data URI) before vector overlays.

### Hotspot Features

- Extracts APD `name` and `region` records into hotspot JSON.
- Falls back to APS geometry-based bounding boxes when explicit region data is absent.

## Scope And Limitations

- The exact supported CGM elements/commands are listed in the Supported CGM Elements And Features section above.
- This project is extraction-oriented: it parses and exports data/SVG/JSON, but does **not** support CGM authoring or round-trip editing.
- Rendering is best-effort for many real-world files; unsupported or profile-specific constructs may be skipped rather than heuristically rewritten.
- Raster composition is metadata-dependent. Clear-text tile arrays are composed directly; binary Cell Array payloads are extracted and used as raster candidates.
- Certain raster decode paths require optional runtime dependencies (`Pillow`, `imagecodecs`). Without them, raw extraction still works but image rendering coverage is reduced.
- JSON exports can be large because they include full element parameter and payload hex data.

## License

python-cgm (C) 2026 Kestin Goforth.

This project is licensed under the BSD 3-Clause License - see the [license file](LICENSE) for details.
