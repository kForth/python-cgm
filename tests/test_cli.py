from __future__ import annotations

__author__ = "Kestin Goforth"
__copyright__ = "Copyright 2026"
__license__ = "BSD-3-Clause"

import json
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


def _apd_property(key: bytes, value: bytes) -> bytes:
    return bytes([len(key)]) + key + bytes([len(value)]) + value


def _make_hotspot() -> bytes:
    begin_name = b"SPOT"
    begin_apd = bytes([len(begin_name)]) + begin_name + b"\x00"
    name_prop = _apd_property(b"name", b"ZONE")
    region_value = (
        b"\x00\x0b\x00\x01\x00\x01\x00\x10\x00\x04"
        + (1).to_bytes(2, "big")
        + (2).to_bytes(2, "big")
        + (3).to_bytes(2, "big")
        + (4).to_bytes(2, "big")
    )
    region_prop = _apd_property(b"region", region_value)

    return (
        _header(0, 21, len(begin_apd))
        + begin_apd
        + _header(9, 1, len(name_prop))
        + name_prop
        + _header(9, 1, len(region_prop))
        + region_prop
        + _header(0, 22, 0)
    )


def test_cli_default_mode_writes_svg(tmp_path: Path) -> None:
    src = tmp_path / "in.cgm"
    out_dir = tmp_path / "out"
    src.write_bytes(_make_cell_array(b"PAY"))

    runner = CliRunner()
    result = runner.invoke(cli, [str(src), str(out_dir)])

    assert result.exit_code == 0
    assert (out_dir / "in_0000.svg").exists()
    assert (out_dir / "in_0000.hotspots.json").exists()


def test_cli_debug_mode_writes_decode_report(tmp_path: Path) -> None:
    src = tmp_path / "in.cgm"
    out_dir = tmp_path / "out"
    hex_payload = "0123456789ABCDEFFEDCBA9876543210"
    src.write_text(
        "BEGTILEARRAY 0 0 0 0 1 1 1 1 0 0 0 0 1 1; "
        f"BITONALTILE 2 16 0 1 '' {hex_payload}; "
        "ENDTILEARRAY;",
        encoding="ascii",
    )

    runner = CliRunner()
    result = runner.invoke(cli, [str(src), str(out_dir), "--debug"])

    assert result.exit_code == 0
    assert (out_dir / "in_0000.svg").exists()
    assert (out_dir / "in_0000.hotspots.json").exists()
    report_path = out_dir / "in_decode_report.json"
    assert report_path.exists()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert "arrays" in report


def test_cli_accepts_clear_text_cgm(tmp_path: Path) -> None:
    src = tmp_path / "in.cgm"
    out_dir = tmp_path / "out"
    src.write_text(
        'VDCEXT 0 0 100 100; LINE 0 0 10 10; BEGAPS "SPOT"; "'
        'APD "name" "ZONE"; APD "region" "1 2 3 4"; ENDAPS;',
        encoding="ascii",
    )

    runner = CliRunner()
    result = runner.invoke(cli, [str(src), str(out_dir)])

    assert result.exit_code == 0
    assert (out_dir / "in_0000.svg").exists()
    hotspot_json = json.loads((out_dir / "in_0000.hotspots.json").read_text(encoding="utf-8"))
    assert hotspot_json
    assert hotspot_json[0]["name"] == "ZONE"
