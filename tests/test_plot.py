import importlib.util
import pytest
from mlglance import plot

plotext_missing = importlib.util.find_spec("plotext") is None


def _rows():
    rows = [{"iter": i, "train_loss": 2.0 - i * 0.01} for i in range(1, 60)]
    rows += [{"iter": i, "val_loss": 1.5} for i in range(20, 60, 20)]
    return rows


def test_ascii_backend_reports_itself_and_honors_height():
    # the watch loop reserves exactly `h` rows for the chart — both backends must deliver h lines
    for h in (6, 10, 18):
        body, backend = plot.loss_plot(_rows(), w=100, h=h, force_ascii=True)
        assert backend == "ascii"
        assert len(body) == h


@pytest.mark.skipif(plotext_missing, reason="plotext not installed")
def test_plotext_backend_selected_and_honors_height():
    # the gap that let the silent ascii-fallback hide: prove plotext is actually used when present
    for h in (6, 10, 18):
        body, backend = plot.loss_plot(_rows(), w=100, h=h, force_ascii=False)
        assert backend == "plotext"
        assert len(body) == h


def test_loss_plot_no_train_data_falls_back_cleanly():
    body, backend = plot.loss_plot([{"iter": 1, "val_loss": 1.5}], w=100, h=10, force_ascii=True)
    assert backend == "ascii"
    assert any("no train_loss" in ln for ln in body)
