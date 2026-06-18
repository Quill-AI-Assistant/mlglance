import pytest
from mlglance.cli import build_parser


def test_cli_defaults():
    a = build_parser().parse_args([])
    assert a.path is None
    assert a.watch is False
    assert a.interval == 5.0
    assert a.total == 0


def test_cli_flags_wire_through():
    a = build_parser().parse_args(
        ["f.jsonl", "--ascii", "--no-color", "--log-y", "--total", "500", "-n", "0.5", "-w"])
    assert a.path == "f.jsonl"
    assert a.ascii and a.no_color and a.log_y and a.watch
    assert a.total == 500
    assert a.interval == 0.5


def test_cli_version_exits_zero(capsys):
    with pytest.raises(SystemExit) as e:
        build_parser().parse_args(["--version"])
    assert e.value.code == 0
    assert "0.1.0" in capsys.readouterr().out
