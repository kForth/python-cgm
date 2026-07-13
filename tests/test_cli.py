from __future__ import annotations

__author__ = "Kestin Goforth"
__copyright__ = "Copyright 2026"
__license__ = "BSD-3-Clause"

import struct
from typing import TYPE_CHECKING

from click.testing import CliRunner

from cgm.cli import cli

if TYPE_CHECKING:
    from pathlib import Path


def _header(class_id: int, element_id: int, length: int) -> bytes:
    value = (class_id << 12) | (element_id << 5) | length
    return value.to_bytes(2, "big")


def _make_cell_array(payload: bytes) -> bytes:
    params = (
        (b"\x00" * 12) + (1).to_bytes(2, "big") + (1).to_bytes(2, "big") + (b"\x00" * 4) + payload
    )
    return _header(4, 9, len(params)) + params


def _make_polyline() -> bytes:
    points = (
        struct.pack(">f", 0.0)
        + struct.pack(">f", 0.0)
        + struct.pack(">f", 10.0)
        + struct.pack(">f", 10.0)
    )
    return _header(4, 1, len(points)) + points


def test_cli_default_mode_writes_bin(tmp_path: Path) -> None:
    src = tmp_path / "in.cgm"
    out_dir = tmp_path / "out"
    src.write_bytes(_make_cell_array(b"PAY"))

    runner = CliRunner()
    result = runner.invoke(cli, [str(src), str(out_dir)])

    assert result.exit_code == 0
    assert (out_dir / "image_0000.bin").exists()


def test_cli_svg_mode_writes_svg(tmp_path: Path) -> None:
    src = tmp_path / "in.cgm"
    out_dir = tmp_path / "out"
    src.write_bytes(_make_polyline())

    runner = CliRunner()
    result = runner.invoke(cli, [str(src), str(out_dir), "--svg"])

    assert result.exit_code == 0
    assert (out_dir / "image_0000.svg").exists()


def test_cli_json_mode_writes_json(tmp_path: Path) -> None:
    src = tmp_path / "in.cgm"
    out_dir = tmp_path / "out"
    src.write_bytes(_make_polyline())

    runner = CliRunner()
    result = runner.invoke(cli, [str(src), str(out_dir), "--json"])

    assert result.exit_code == 0
    assert (out_dir / "image_0000.json").exists()


def test_cli_svg_and_json_mode_writes_both_files(tmp_path: Path) -> None:
    src = tmp_path / "in.cgm"
    out_dir = tmp_path / "out"
    src.write_bytes(_make_polyline())

    runner = CliRunner()
    result = runner.invoke(cli, [str(src), str(out_dir), "--svg", "--json"])

    assert result.exit_code == 0
    assert (out_dir / "image_0000.svg").exists()
    assert (out_dir / "image_0000.json").exists()
