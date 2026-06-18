"""Plot backends: plotext (primary) with a zero-dep ASCII fallback."""
from __future__ import annotations

from . import theme
from .metrics import ema
from .theme import dim, cyan, bold, yellow

try:
    import plotext as _plt
    HAVE_PLOTEXT = True
except ImportError:  # pragma: no cover
    HAVE_PLOTEXT = False


def have_plotext() -> bool:
    return HAVE_PLOTEXT


def _plotext_plot(tr, ema_pts, val_pts, best_val, w, h, log_y):
    _plt.clf()
    _plt.theme("clear")
    _plt.limit_size(False, False)   # honour our computed canvas; don't re-clamp to the tty
    _plt.plotsize(w, h)
    if tr:
        xs, ys = zip(*tr)
        _plt.scatter(list(xs), list(ys), marker=theme.TRAIN_MARKER, color=theme.TRAIN_COLOR)
    if ema_pts:
        xs, ys = zip(*ema_pts)
        _plt.plot(list(xs), list(ys), marker=theme.EMA_MARKER, color=theme.EMA_COLOR)
    if val_pts:
        xs, ys = zip(*val_pts)
        _plt.scatter(list(xs), list(ys), marker=theme.VAL_MARKER, color=theme.VAL_COLOR)
    if best_val is not None:
        _plt.horizontal_line(best_val, color=theme.BEST_COLOR)
    if log_y:
        _plt.yscale("log")
    _plt.xlabel("iter")
    lines = _plt.build().split("\n")
    return [ln for ln in lines if ln != ""]


def _ascii_plot(tr, ema_pts, val_pts, w, h, log_y):
    if not tr:
        return ["  (no train_loss yet)"]
    import math
    tf = (lambda v: math.log10(max(v, 1e-9))) if log_y else (lambda v: v)

    all_its = [i for i, _ in tr] + [i for i, _ in val_pts]
    all_ls  = [tf(v) for _, v in tr] + [tf(v) for _, v in val_pts]
    i0, i1  = min(all_its), max(all_its)
    l0, l1  = min(all_ls),  max(all_ls)
    if i1 == i0:       i1 = i0 + 1
    if l1 - l0 < 1e-9: l1 = l0 + 1

    gh = max(1, h - 2)               # reserve 2 rows for the axis + tick line → total height == h
    gw = max(10, w - 9)              # reserve 9 cols for the y-axis gutter → total width == w
    grid = [[" "] * gw for _ in range(gh)]

    def place(it, val, ch):
        x = max(0, min(gw - 1, int((it - i0) / (i1 - i0) * (gw - 1))))
        y = max(0, min(gh - 1, int((l1 - tf(val)) / (l1 - l0) * (gh - 1))))
        grid[y][x] = ch

    for it, v in tr:       place(it, v, ".")
    for it, v in ema_pts:  place(it, v, "~")
    for it, v in val_pts:  place(it, v, "o")

    paint = {".": dim("."), "~": cyan("─"), "o": bold(yellow("o"))} if theme.USE_COLOR \
        else {".": ".", "~": "─", "o": "o"}
    bar, cor = dim("│"), dim("└")
    span = gh - 1 or 1
    out = []
    for r, row in enumerate(grid):
        v = l1 - (r / span) * (l1 - l0)
        v = (10 ** v) if log_y else v
        g = f"{v:7.3f} {bar}" if r % 2 == 0 else f"        {bar}"
        out.append(g + "".join(paint.get(c, c) for c in row))
    out.append("        " + cor + dim("─" * gw))               # 9-col gutter + gw == w
    axis = f"{i0}{' ' * max(1, gw - len(str(i0)) - len(str(i1)))}{i1}"
    out.append(dim("         " + axis[:gw]))                   # iter axis, fits within w
    return out


def loss_plot(rows, w, h, log_y=False, force_ascii=False):
    """Render the loss curve. Returns (lines, backend_used) so the caller can
    report the backend that actually drew — not the one it hoped for (a plotext
    failure silently degrades to ascii, and the title must not keep saying 'plotext')."""
    tr      = [(r["iter"], r["train_loss"]) for r in rows if "train_loss" in r]
    val_pts = [(r["iter"], r["val_loss"])   for r in rows if "val_loss" in r]
    ema_pts = list(zip([i for i, _ in tr], ema([v for _, v in tr]))) if tr else []
    best_val = min((v for _, v in val_pts), default=None)

    if HAVE_PLOTEXT and not force_ascii:
        try:
            return _plotext_plot(tr, ema_pts, val_pts, best_val, w, h, log_y), "plotext"
        except Exception:
            pass   # plotext blew up at this size/data — degrade to ascii, reported honestly
    return _ascii_plot(tr, ema_pts, val_pts, w, h, log_y), "ascii"
