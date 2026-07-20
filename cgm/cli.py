"""Command-line interface for final SVG and hotspot export."""

from __future__ import annotations

__author__ = "Kestin Goforth"
__copyright__ = "Copyright 2026"
__license__ = "BSD-3-Clause"

import logging
from pathlib import Path

import click

from cgm.extract import (
    extract_hotspots_to_directory,
    extract_rendered_images_to_directory,
)


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument("srcfile", type=click.Path(exists=True, dir_okay=False, path_type=Path))
@click.argument("output_dir", type=click.Path(file_okay=False, path_type=Path))
@click.option(
    "--debug",
    is_flag=True,
    help="Enable debug logging and write decode diagnostics report.",
)
def cli(
    srcfile: Path,
    output_dir: Path,
    debug: bool,
) -> None:
    """Export final SVG and hotspot JSON using the source filename as output stem."""
    if debug:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(levelname)s:%(name)s:%(message)s",
        )

    written_paths: list[Path] = []
    stem = srcfile.stem or "image"

    try:
        written_paths.extend(
            extract_rendered_images_to_directory(
                srcfile,
                output_dir,
                stem=stem,
                debug_report=debug,
            )
        )
        hotspot_path = extract_hotspots_to_directory(srcfile, output_dir, stem=stem)
        written_paths.append(hotspot_path)
    except RuntimeError as exc:
        raise click.ClickException(str(exc)) from exc

    for path in written_paths:
        click.echo(str(path))


def main() -> None:
    """Run the Click command entrypoint."""
    cli.main(standalone_mode=True)


if __name__ == "__main__":
    main()
