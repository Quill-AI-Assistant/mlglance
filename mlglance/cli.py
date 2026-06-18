"""Command-line entry point + the watch loop."""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time

from . import __version__, theme
from .render import render
from .theme import dim

_CLEAR = "cls" if os.name == "nt" else "clear"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="mlglance",
        description="Terminal-native live training monitor (loss curves + verdict).")
    p.add_argument("path", nargs="?", default=None,
                   help="metrics JSONL to watch (or 'demo' for a live demo)")
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("-w", "--watch", action="store_true", help="refresh continuously")
    p.add_argument("-n", "--interval", type=float, default=5, help="refresh seconds (default 5, sub-second ok)")
    p.add_argument("--total", type=int, default=0, help="total steps (else auto-detected)")
    p.add_argument("--title", default=None, help="dashboard title (else from path)")
    p.add_argument("--log-y", action="store_true", help="log-scale loss axis")
    p.add_argument("--ascii", action="store_true", help="force the zero-dep ascii backend")
    p.add_argument("--no-color", action="store_true", help="disable ANSI colour")
    p.add_argument("--width", type=int, default=0, help="override canvas width")
    p.add_argument("--height", type=int, default=0, help="override canvas height")
    p.add_argument("--demo", action="store_true", help="run a live synthetic demo")
    p.add_argument("--scenario", default="overfit",
                   choices=["converge", "overfit", "diverge"], help="demo scenario")
    return p


def _clear():
    # ANSI home+erase: atomic, no per-cycle fork, less flicker. Fall back to the
    # clear/cls binary only when stdout isn't a tty (the ANSI codes would be noise).
    if sys.stdout.isatty():
        sys.stdout.write("\033[H\033[2J")
    else:
        os.system(_CLEAR)


def watch_loop(path, args):
    last_key, cached = None, ""
    try:
        while True:
            try:
                mtime = os.stat(path).st_mtime
            except OSError:
                mtime = None
            key = (mtime, shutil.get_terminal_size())
            if key != last_key:
                try:
                    cached = render(path, args.total, args.title, args.width, args.height,
                                    args.log_y, args.ascii)
                    last_key = key
                except Exception as e:        # one bad frame must not kill a live watch — retry next tick
                    cached = f"  [mlglance: render error — {e}]\n  retrying…"
            _clear()
            print(cached)
            print(dim(f"\n  watching every {args.interval}s — Ctrl-C to stop"))
            time.sleep(args.interval)
    except KeyboardInterrupt:
        pass


def main(argv=None):
    args = build_parser().parse_args(argv)
    theme.enabled(sys.stdout.isatty() and not args.no_color)

    if args.demo or args.path == "demo":
        from .demo import run_demo
        run_demo(args)
        return

    path = args.path or "metrics.jsonl"
    if args.watch:
        watch_loop(path, args)
    else:
        print(render(path, args.total, args.title, args.width, args.height,
                     args.log_y, args.ascii))


if __name__ == "__main__":
    main()
