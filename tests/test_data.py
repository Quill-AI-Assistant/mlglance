from mlglance.data import normalize, series


def test_alias_normalization():
    raw = [{"step": 5, "loss": 1.2, "learning_rate": 1e-4, "eval_loss": 1.5}]
    out = normalize(raw)
    assert out[0]["iter"] == 5
    assert out[0]["train_loss"] == 1.2
    assert out[0]["lr"] == 1e-4
    assert out[0]["val_loss"] == 1.5


def test_iter_synthesized_when_absent():
    raw = [{"loss": 1.0}, {"loss": 0.9}]
    out = normalize(raw)
    assert [r["iter"] for r in out] == [1, 2]


def test_iter_numbering_with_interspersed_non_dict_rows():
    # a stray non-dict JSON line must not shift the synthesized iter of real rows
    raw = [["junk"], {"loss": 0.5}, 42, {"loss": 0.4}]
    out = normalize(raw)
    assert [r["iter"] for r in out] == [1, 2]
    assert [r["train_loss"] for r in out] == [0.5, 0.4]


def test_non_numeric_and_nonfinite_values_are_dropped_not_crashed():
    raw = [{"iter": 1, "loss": "NaN", "val_loss": "oops"},
           {"iter": 2, "loss": 0.3, "lr": float("inf")}]
    out = normalize(raw)
    assert "train_loss" not in out[0]      # "NaN" string dropped
    assert "val_loss" not in out[0]        # non-numeric dropped
    assert out[1]["train_loss"] == 0.3
    assert "lr" not in out[1]              # inf dropped


def test_iter_kept_as_int():
    out = normalize([{"step": 50, "loss": 1.0}])
    assert out[0]["iter"] == 50 and isinstance(out[0]["iter"], int)


def test_series_skips_missing():
    rows = [{"iter": 1, "train_loss": 1.0}, {"iter": 2}, {"iter": 3, "train_loss": 0.8}]
    xs, ys = series(rows, "train_loss")
    assert xs == [1, 3]
    assert ys == [1.0, 0.8]
