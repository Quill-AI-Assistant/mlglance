from mlglance.metrics import ema, lin_slope, detect_val_every, verdict


def test_ema_tracks_and_smooths():
    assert ema([]) == []
    out = ema([1.0, 1.0, 1.0])
    assert all(abs(v - 1.0) < 1e-9 for v in out)
    # EMA of a falling series stays above the latest raw point (lags)
    out = ema([5, 4, 3, 2, 1])
    assert out[-1] > 1.0


def test_lin_slope_sign():
    assert lin_slope([5, 4, 3, 2, 1]) < 0
    assert lin_slope([1, 2, 3, 4, 5]) > 0


def test_lin_slope_flat_and_short():
    assert lin_slope([2.0] * 20) == 0.0     # flat series → zero slope (plateau)
    assert lin_slope([1.0, 2.0]) == 0.0     # len < 3 guard


def test_detect_val_every():
    assert detect_val_every([25, 50, 75, 100]) == 25
    assert detect_val_every([50, 100, 150]) == 50
    assert detect_val_every([10]) is None


def test_detect_val_every_lower_median_on_even_lengths():
    # one late eval gives gaps [100, 150]; the cadence is 100, not the upper-median 150
    assert detect_val_every([0, 100, 250]) == 100


def _ema_for(iters, vals):
    return ema(vals)


def test_verdict_converging_when_val_improving():
    tr_i = list(range(1, 21))
    tr_v = [5 - i * 0.2 for i in tr_i]
    val = [(5, 3.0), (10, 2.5), (15, 2.0), (20, 1.6)]   # latest is best
    sev, head, _ = verdict(tr_i, ema(tr_v), val, 1.6, 20, 20, 5)
    assert sev == "good"
    assert "converging" in head


def test_verdict_overfitting_when_train_down_val_up():
    tr_i = list(range(1, 101))
    tr_v = [3.0 * (0.97 ** i) + 0.3 for i in tr_i]     # train keeps falling
    # val bottoms at iter 40 then drifts up
    val = [(20, 2.0), (40, 1.5), (60, 1.7), (80, 1.9), (100, 2.0)]
    sev, head, detail = verdict(tr_i, ema(tr_v), val, 1.5, 40, 100, 20)
    assert sev == "warn"
    assert "overfitting" in head
    assert "iter 40" in detail


def test_verdict_no_val_falls_back_to_train():
    tr_i = list(range(1, 21))
    tr_v = [5 - i * 0.2 for i in tr_i]
    sev, head, _ = verdict(tr_i, ema(tr_v), [], None, None, 20, None)
    assert sev == "good" and "converging" in head


def test_verdict_no_val_train_rising_is_bad():
    tr_i = list(range(1, 21))
    tr_ema = [0.1 * i for i in tr_i]                    # train EMA climbing
    sev, head, _ = verdict(tr_i, tr_ema, [], None, None, 20, None)
    assert sev == "bad" and "train rising" in head


def test_verdict_no_val_flat_is_plateau():
    tr_i = list(range(1, 21))
    sev, head, _ = verdict(tr_i, [2.0] * 20, [], None, None, 20, None)
    assert sev == "warn" and "plateaued" in head


def test_verdict_diverging_when_train_and_val_both_rise():
    tr_i = list(range(1, 101))
    tr_ema = [1.0 + 0.02 * k for k in range(100)]       # train EMA rose well past its val-best level
    val = [(20, 1.5), (40, 1.7), (60, 1.9), (80, 2.1), (100, 2.3)]   # val worsening since iter 20
    sev, head, detail = verdict(tr_i, tr_ema, val, 1.5, 20, 100, 20)
    assert sev == "bad"
    assert "DIVERGING" in head
    assert "stopping" in detail


def test_verdict_val_only_does_not_crash():
    # verdict is a public fn; a val-only run (empty tr_iters) on the diverging path must not raise
    sev, head, _ = verdict([], [], [(100, 2.0), (200, 2.5)], 2.0, 100, 200, 100)
    assert sev == "warn" and "val rising" in head
