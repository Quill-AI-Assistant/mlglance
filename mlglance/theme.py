"""ANSI colour helpers + the shared palette."""
from __future__ import annotations

USE_COLOR = True


def enabled(flag: bool) -> None:
    global USE_COLOR
    USE_COLOR = flag


def _c(code: str, text: object) -> str:
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else str(text)


def bold(t):   return _c("1",  t)
def dim(t):    return _c("2",  t)
def red(t):    return _c("31", t)
def green(t):  return _c("32", t)
def yellow(t): return _c("33", t)
def blue(t):   return _c("34", t)
def cyan(t):   return _c("36", t)
def white(t):  return _c("97", t)

TRAIN_COLOR = "white"
EMA_COLOR   = "cyan"
VAL_COLOR   = "orange"
BEST_COLOR  = "green"

# plot markers — kept here so the dashboard's "look" lives in one place.
TRAIN_MARKER = "·"        # small middle-dot: the noisy raw signal, kept light
VAL_MARKER   = "●"        # bold eval points: sparse + important, meant to stand out
EMA_MARKER   = "braille"  # thin connected braille line: the trend, the chart's spine
