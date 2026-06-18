"""Derived signals + the training verdict."""
from __future__ import annotations

import re
import subprocess

EMA_SPAN = 10


def mean(xs: list) -> float:
    return sum(xs) / len(xs) if xs else float("nan")


def ema(vals: list, span: int = EMA_SPAN) -> list:
    if not vals:
        return []
    a = 2.0 / (span + 1)
    out = [vals[0]]
    for v in vals[1:]:
        out.append(a * v + (1 - a) * out[-1])
    return out


def lin_slope(vals: list, n: int = 20) -> float:
    v = vals[-n:]
    if len(v) < 3:
        return 0.0
    xs = list(range(len(v)))
    xm, ym = mean(xs), mean(v)
    num = sum((x - xm) * (y - ym) for x, y in zip(xs, v))
    den = sum((x - xm) ** 2 for x in xs)
    return num / den if den else 0.0


def detect_val_every(val_iters: list):
    if len(val_iters) < 2:
        return None
    diffs = sorted(b - a for a, b in zip(val_iters, val_iters[1:]) if b > a)
    # lower median: for an even-length list take the smaller middle gap, so a single
    # late eval can't inflate the detected cadence (which would mask a stale checkpoint)
    return diffs[(len(diffs) - 1) // 2] if diffs else None


_TOTAL_RE = re.compile(
    r"--(?:max[-_]?steps|total[-_]?steps|train[-_]?steps|num[-_]?steps|steps)[= ]+(\d+)")
_TOTAL_CACHE = {"v": 0}


def detect_total() -> int:
    if _TOTAL_CACHE["v"]:
        return _TOTAL_CACHE["v"]
    try:
        out = subprocess.run(["ps", "-axww", "-o", "command"],
                             capture_output=True, text=True, timeout=2).stdout
    except Exception:
        return 0
    for line in out.splitlines():
        low = line.lower()
        if "mlglance" in low or "dashboard" in low or "train" not in low:
            continue
        m = _TOTAL_RE.search(line)
        if m:
            _TOTAL_CACHE["v"] = int(m.group(1))
            return _TOTAL_CACHE["v"]
    return 0


def verdict(tr_iters, tr_ema, val_pts, best_val, best_it, cur_iter, val_every):
    if len(val_pts) < 2 or best_val is None:
        sl = lin_slope(tr_ema, 20)
        if sl < -0.0015:
            return "good", f"converging  (train EMA {sl:+.4f}/step)", None
        if sl > 0.0015:
            return "bad", f"train rising  ({sl:+.4f}/step)", None
        return "warn", f"plateaued  (train EMA {sl:+.4f}/step)", None

    latest_val = val_pts[-1][1]
    delta      = latest_val - best_val
    since_best = cur_iter - best_it
    stale      = bool(val_every) and since_best >= 2 * val_every

    if delta <= 0.01:
        return "good", f"converging  (val {latest_val:.4f}, at/near best)", None

    if not tr_iters:        # val-only run (verdict called directly): no train signal to compare
        return "warn", f"val rising  (+{delta:.3f} above best @ {best_it})", None

    bi      = min(range(len(tr_iters)), key=lambda k: abs(tr_iters[k] - best_it))
    at_best = mean(tr_ema[max(0, bi - 5):bi + 5] or tr_ema[:1])
    now_lvl = mean(tr_ema[-10:])
    train_gain = at_best - now_lvl

    best_note  = f"best (val-optimal) ckpt @ iter {best_it}"
    stale_note = f"  ·  val {since_best} steps stale" if stale else ""

    if train_gain > 0.05:
        return ("warn",
                f"overfitting  (train↓{train_gain:.2f} since best / val↑ +{delta:.3f})",
                best_note + " — verify latest vs it" + stale_note)
    if train_gain < -0.05:
        return ("bad",
                f"DIVERGING  (train↑{-train_gain:.2f} & val↑ +{delta:.3f}) — instability",
                best_note + "; consider stopping / lowering LR")
    return ("warn",
            f"plateaued  (train flat / val +{delta:.3f} above best @ {best_it})",
            (best_note + stale_note) if stale else None)
