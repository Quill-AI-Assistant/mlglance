"""Assemble the full dashboard string and fit it to the terminal.

Layout is three labelled sections stacked above the chart:
  ── LOSS ──        train + EMA + verdict, val/best, the val-optimal note
  ── RUN ──         progress/ETA, then lr + step-time + throughput + RAM + grad + health
  ── LOSS CURVE ──  a divider that doubles as the chart legend
The chart then grows to fill every remaining row.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path
from types import SimpleNamespace

from . import data, plot, theme
from .metrics import (ema, mean, detect_val_every, detect_total, verdict)
from .theme import bold, dim, red, green, yellow, cyan, white

FOOTER_RESERVE = 3                      # watch loop's blank + "watching …" line, plus 1 row of scroll headroom
SEV_COLOR = {"good": green, "warn": yellow, "bad": red}    # the only verdicts metrics.verdict() emits
SEV_SYM   = {"good": "↓",   "warn": "⚠",    "bad": "↑↑"}

# _vlen / _ANSI assume SGR-only colour codes — true for everything theme.py emits (the only producer).
_ANSI = re.compile(r"\033\[[0-9;]*m")


def _vlen(s: str) -> int:
    """Visible width of a string, ignoring ANSI colour codes."""
    return len(_ANSI.sub("", s))


def _vrows(line: str, width: int) -> int:
    """How many terminal rows a line occupies once wrapped at `width`."""
    return max(1, -(-_vlen(line) // width))


def _div(label: str, width: int, legend: str = "") -> str:
    """A titled section divider, optionally carrying an inline legend:
       ── LABEL ─────────…        or        ── LABEL ── <legend> ────…"""
    head = dim("── ") + bold(label) + (dim(" ── ") if legend else dim(" "))
    body = (legend + " ") if legend else ""
    fill = max(0, width - _vlen(head) - _vlen(body))
    return head + body + dim("─" * fill)


def default_title(path: str) -> str:
    p = Path(path).resolve()
    name = p.parent.parent.name if p.parent.name == "data" else p.parent.name
    return name or "run"


def fmt_eta(seconds: float) -> str:
    if seconds < 1:
        return "done"
    h, r = divmod(int(seconds), 3600)
    m, s = divmod(r, 60)
    return f"{h}h {m:02d}m" if h else f"{m}m {s:02d}s"


def lr_phase(lrs: list) -> str:
    if len(lrs) < 3:
        return "warmup"
    d = lrs[-1] - lrs[-min(5, len(lrs))]
    if d > 1e-9:
        return "warmup"
    return "floor" if abs(d) < 1e-9 else "cosine decay"


def progress_line(done, total, rate_s, auto):
    width = 42
    if total and done <= total:
        frac = done / total
        bar = "█" * int(frac * width) + "░" * (width - int(frac * width))
        eta = fmt_eta(max(0, total - done) * rate_s) if rate_s else "—"
        tag = dim(" auto") if auto else ""
        return (f"  progress  {cyan('[' + bar + ']')}  "
                f"{bold(f'{done}/{total}')}{tag}  {frac*100:.0f}%   ETA {bold(eta)}")
    note = f"+{done-total} past total {total}" if total else "no --total set"
    return f"  progress  step {bold(str(done))}   {yellow(note)} {dim('— --total N for %/ETA')}"


def _prepare(rows, total_steps, auto_total) -> SimpleNamespace:
    """Reduce the metric rows to the handful of signals the sections render."""
    last = rows[-1]
    tr_iters, tr_vals = data.series(rows, "train_loss")
    val_pts = list(zip(*data.series(rows, "val_loss")))   # [] when no val rows
    lrs   = data.series(rows, "lr")[1]
    steps = data.series(rows, "step_s")[1]
    peaks = data.series(rows, "peak_gb")[1]
    grads = data.series(rows, "grad_norm")[1]
    toks  = data.series(rows, "tokens")[1]

    tr_ema    = ema(tr_vals) if tr_vals else []
    val_every = detect_val_every([i for i, _ in val_pts])
    best_val  = min((v for _, v in val_pts), default=None)
    best_it   = next((i for i, v in val_pts if v == best_val), None)
    mean_step = mean(steps) if steps else 0
    rate_s    = mean(steps[-10:]) if steps else None

    if auto_total and not total_steps:
        total_steps = detect_total()

    return SimpleNamespace(
        cur_iter=last["iter"], total_steps=total_steps, auto_total=auto_total,
        tr_iters=tr_iters, tr_vals=tr_vals, tr_ema=tr_ema,
        val_pts=val_pts, val_every=val_every, best_val=best_val, best_it=best_it,
        lrs=lrs, steps=steps, peaks=peaks, grads=grads, toks=toks,
        mean_step=mean_step, rate_s=rate_s,
        creep=(rate_s / mean_step) if (rate_s and mean_step) else 1.0,
        maxpeak=max(peaks) if peaks else 0,
    )


def _loss_section(m, width) -> list:
    if not (m.tr_vals or m.val_pts):
        return []
    out = [_div("LOSS", width)]

    if m.tr_vals:
        sev, head, detail = verdict(m.tr_iters, m.tr_ema, m.val_pts,
                                    m.best_val, m.best_it, m.cur_iter, m.val_every)
        out.append(f"  train  {bold(f'{m.tr_vals[-1]:.4f}')}   "
                   f"{cyan(f'EMA {m.tr_ema[-1]:.4f}')}   "
                   f"{SEV_COLOR[sev](SEV_SYM[sev] + ' ' + head)}")
    else:
        detail = None

    if m.val_pts:
        latest = m.val_pts[-1][1]
        since  = m.cur_iter - m.best_it
        cad    = f"/{m.val_every}" if m.val_every else ""
        nv     = (((m.cur_iter // m.val_every) + 1) * m.val_every) if m.val_every else None
        nxt    = f" · next ~{nv}" if (nv and (not m.total_steps or nv <= m.total_steps)) else ""
        col    = green if since == 0 else (yellow if (m.val_every and since >= 2*m.val_every) else dim)
        out.append(f"  val    {bold(f'{latest:.4f}')}{dim(cad)}  "
                   f"best={green(f'{m.best_val:.4f}')} @ {m.best_it}  "
                   f"{col(f'({since} since best{nxt})')}")

    if detail:
        out.append(dim("         " + detail))
    return out


def _run_section(m, width) -> list:
    out = [_div("RUN", width),
           progress_line(m.cur_iter, m.total_steps, m.rate_s,
                         m.auto_total and bool(m.total_steps))]

    parts = []
    if m.lrs:
        parts.append(f"lr {m.lrs[-1]:.2e} {dim('[' + lr_phase(m.lrs) + ']')}")
    if m.steps:
        parts.append(f"step {m.steps[-1]:.1f}s {dim(f'(mean {m.mean_step:.1f}s ×{m.creep:.2f})')}")
        if m.toks and m.steps[-1] > 0:
            parts.append(dim(f"{m.toks[-1]/m.steps[-1]:,.0f} tok/s"))
    if m.peaks:
        parts.append(f"peak {m.peaks[-1]:.2f} GB")
    if m.grads:
        parts.append(f"grad {m.grads[-1]:.2f}")

    if parts:
        line = "  " + "   ".join(parts)
        if m.steps or m.peaks:
            health = (yellow("WATCH (step creep)") if (m.steps and m.creep >= 1.4)
                      else yellow("WATCH (peak RAM)") if (m.peaks and m.maxpeak >= 12)
                      else green("HEALTHY"))
            line += f"   → {health}"
        out.append(line)
    return out


def _legend(log_y) -> str:
    return (white("·") + dim(" train") + "  " + yellow("● val") + "  "
            + cyan("─ EMA") + "  " + green("─ best")
            + (dim("   (log-y)") if log_y else ""))


def render(path, total_steps=0, title=None, cols=0, rows_=0, log_y=False, force_ascii=False):
    rows = data.normalize(data.load(path))
    rows = [r for r in rows if "train_loss" in r or "val_loss" in r] or rows

    term_c, term_r = shutil.get_terminal_size((100, 30))
    term_c, term_r = (cols or term_c), (rows_ or term_r)
    plot_w = max(50, min(term_c - 2, 240))

    title = title or default_title(path)
    auto_total = not total_steps

    # colour ⇒ plotext (its glyphs are coloured); no-colour/pipe ⇒ clean ascii
    eff_ascii = force_ascii or not theme.USE_COLOR
    title_head = bold(f"  {title}") + f"    {dim(path)}"

    def _title(backend: str) -> str:
        hint = ""
        if backend == "ascii" and not eff_ascii:        # we wanted the rich chart but got ascii
            hint = (dim("  · plotext not installed (pip install plotext)")
                    if not plot.have_plotext()
                    else dim("  · plotext failed — using ascii"))
        return f"{title_head}  {dim('· ' + backend)}{hint}"

    if not rows:
        note = "" if Path(path).exists() else dim("  (file not found)")
        return "\n".join([_title("plotext" if (plot.have_plotext() and not eff_ascii) else "ascii"),
                          "  (no metrics yet — waiting for first step)" + note])

    m = _prepare(rows, total_steps, auto_total)

    pre = ([_title("plotext" if (plot.have_plotext() and not eff_ascii) else "ascii"), ""]
           + _loss_section(m, term_c) + [""]
           + _run_section(m, term_c) + [""]
           + [_div("LOSS CURVE", term_c, _legend(log_y))])

    # count wrapped rows, not list items, so a long sysline/legend at a narrow width can't steal chart rows
    used = sum(_vrows(line, term_c) for line in pre)
    plot_h = max(6, term_r - used - FOOTER_RESERVE)     # both backends emit exactly plot_h lines
    body, used_backend = plot.loss_plot(rows, w=plot_w, h=plot_h, log_y=log_y, force_ascii=eff_ascii)
    pre[0] = _title(used_backend)                       # reflect the backend that actually drew

    return "\n".join(pre + body)
