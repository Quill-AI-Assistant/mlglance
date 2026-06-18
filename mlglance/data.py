"""Read + normalise a metrics JSONL stream."""
from __future__ import annotations

import json
import math

KEY_ALIASES = {
    "iter":       ("iter", "step", "global_step", "steps", "i", "epoch"),
    "train_loss": ("train_loss", "loss", "train", "training_loss"),
    "val_loss":   ("val_loss", "val", "eval_loss", "valid_loss", "validation_loss"),
    "lr":         ("lr", "learning_rate"),
    "peak_gb":    ("peak_gb", "peak_mem", "peak_mem_gb", "mem_gb", "gpu_gb"),
    "step_s":     ("step_s", "step_time", "step_time_s", "sec_per_step", "dt", "elapsed_s"),
    "grad_norm":  ("grad_norm", "gradnorm", "grad", "gnorm"),
    "tokens":     ("tokens", "tok", "num_tokens", "n_tokens"),
}


def load(path: str) -> list:
    rows = []
    try:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except OSError:        # missing, permission-denied, is-a-dir, NFS hiccup — treat as "no data yet"
        pass
    return rows


def normalize(rows: list) -> list:
    out = []
    kept = 0                       # counts only dict rows → stable synthesized iter
    for r in rows:
        if not isinstance(r, dict):
            continue
        kept += 1
        nr = {}
        for canon, aliases in KEY_ALIASES.items():
            for a in aliases:
                if a in r and r[a] is not None:
                    try:
                        val = float(r[a])      # tolerate int/str numbers
                    except (TypeError, ValueError):
                        continue               # non-numeric → try next alias / skip
                    if not math.isfinite(val):
                        continue               # drop NaN/inf rather than poison min()/plot
                    nr[canon] = int(val) if (canon == "iter" and val == int(val)) else val
                    break
        nr.setdefault("iter", kept)
        out.append(nr)
    return out


def series(rows: list, key: str):
    xs, ys = [], []
    for r in rows:
        if key in r and "iter" in r:
            xs.append(r["iter"])
            ys.append(r[key])
    return xs, ys
