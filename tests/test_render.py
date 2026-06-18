import json
from mlglance import theme
from mlglance.render import render, fmt_eta, lr_phase, default_title, progress_line

theme.enabled(False)  # deterministic, no ANSI in assertions


def _write(tmp_path, scenario):
    import math
    p = tmp_path / "m.jsonl"
    rows = []
    for i in range(1, 121):
        r = {"iter": i, "train_loss": 2.0 * math.exp(-i / 40) + 0.3,
             "lr": 1e-4, "step_s": 0.2, "peak_gb": 7.0}
        if i % 20 == 0:
            r["val_loss"] = 1.5 * math.exp(-i / 50) + 0.4
        rows.append(r)
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return str(p)


def test_render_fits_height(tmp_path):
    path = _write(tmp_path, "converge")
    for h in (22, 24, 28, 40, 52):       # incl. small ssh/tmux panes
        out = render(path, total_steps=120, cols=110, rows_=h, force_ascii=True)
        n = out.count("\n") + 1
        # render + 2 watch-footer lines must fit the terminal height
        assert n + 2 <= h, f"height {h}: emitted {n} lines"


def test_render_has_key_panels(tmp_path):
    path = _write(tmp_path, "converge")
    out = render(path, total_steps=120, cols=110, rows_=40, force_ascii=True)
    assert "LOSS" in out and "RUN" in out            # labelled sections
    assert "progress" in out
    assert "train" in out and "val" in out
    assert "converging" in out  # the fixture is a converging run — pin the exact verdict
    assert "120" in out  # total in progress line


def test_render_empty_file(tmp_path):
    p = tmp_path / "empty.jsonl"
    p.write_text("")
    out = render(str(p), cols=80, rows_=30)
    assert "waiting for first step" in out


def test_fmt_eta():
    assert fmt_eta(0) == "done"
    assert fmt_eta(0.5) == "done"          # sub-second ⇒ done, not a misleading "0m 00s"
    assert fmt_eta(59) == "0m 59s"
    assert fmt_eta(3661) == "1h 01m"


def test_lr_phase():
    assert lr_phase([1e-5, 2e-5]) == "warmup"                          # len < 3
    assert lr_phase([1e-5, 5e-5, 9e-5]) == "warmup"                    # still rising
    assert lr_phase([1e-4] * 6) == "floor"                             # flat
    assert lr_phase([1e-4, 8e-5, 6e-5, 4e-5, 2e-5, 1e-5]) == "cosine decay"


def test_default_title(tmp_path):
    assert default_title(str(tmp_path / "myrun" / "data" / "m.jsonl")) == "myrun"
    assert default_title(str(tmp_path / "runB" / "m.jsonl")) == "runB"


def test_progress_line_overrun():
    assert "+30 past total 120" in progress_line(150, 120, 0.2, False)
    assert "no --total set" in progress_line(50, 0, 0.2, False)


def test_render_val_only_data(tmp_path):
    p = tmp_path / "v.jsonl"
    rows = [{"iter": i, "val_loss": 1.5 - i * 0.001} for i in range(20, 120, 20)]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    out = render(str(p), cols=100, rows_=30, force_ascii=True)
    assert "val" in out and "best=" in out
    assert "no train_loss yet" in out      # chart falls back gracefully with no train series


def test_render_train_only_data(tmp_path):
    p = tmp_path / "t.jsonl"
    rows = [{"iter": i, "train_loss": 2.0 - i * 0.01} for i in range(1, 60)]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    out = render(str(p), cols=100, rows_=30, force_ascii=True)
    assert "train" in out and "converging" in out
    assert "best=" not in out               # no val rows ⇒ no best-val line
