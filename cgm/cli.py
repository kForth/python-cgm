"""Command-line interface for python-cgm."""

from __future__ import annotations

__author__ = "Kestin Goforth"
__copyright__ = "Copyright 2026"
__license__ = "BSD-3-Clause"

import logging
from pathlib import Path

import click

from .extract import (
    extract_data_json_to_directory,
    extract_raw_images_to_directory,
    extract_vector_svg_to_directory,
)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("srcfile", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("output_dir", type=click.Path(file_okay=False, path_type=Path))
@click.option(
    "--stem",
    default="image",
    show_default=True,
    help="Filename stem for extracted payload files.",
)
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug logging for CGM parsing and extraction.",
)
@click.option(
    "--svg",
    is_flag=True,
    help="Convert vector CGM drawing commands into a best-effort SVG.",
)
@click.option(
    "--json",
    "emit_json",
    is_flag=True,
    help="Export parsed/stored CGM data and metadata as JSON.",
)
def cli(
    srcfile: Path, output_dir: Path, stem: str, debug: bool, svg: bool, emit_json: bool
) -> None:
    """Extract raw payloads and/or export SVG/JSON from SRCFILE into OUTPUT_DIR."""
    if debug:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(levelname)s:%(name)s:%(message)s",
        )

    written_paths: list[Path] = []

    if svg:
        svg_path = extract_vector_svg_to_directory(srcfile, output_dir, stem=stem)
        written_paths.append(svg_path)

    if emit_json:
        json_path = extract_data_json_to_directory(srcfile, output_dir, stem=stem)
        written_paths.append(json_path)

    if not svg and not emit_json:
        written_paths.extend(extract_raw_images_to_directory(srcfile, output_dir, stem=stem))

    click.echo(f"Extracted {len(written_paths)} image payload(s).")
    for path in written_paths:
        click.echo(str(path))


def main() -> None:
    """Run the Click command entrypoint."""
    cli.main(standalone_mode=True)


if __name__ == "__main__":
    main()
