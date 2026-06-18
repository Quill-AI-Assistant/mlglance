# Changelog

All notable changes to mlglance are documented here.
This project adheres to [Semantic Versioning](https://semver.org/) and the
[Keep a Changelog](https://keepachangelog.com/) format.

## [0.1.0] — 2026-06-18

### Added
- Initial release.
- Tail-watching of a metrics JSONL with auto-detected key aliases
  (`step`/`global_step`, `loss`/`train_loss`, `eval_loss`/`val_loss`, …),
  eval cadence, and total step count.
- Loss-curve chart: plotext braille backend with a zero-dependency ASCII
  fallback; the dashboard reports which backend actually drew.
- Validation-driven verdict (`converging` / `overfitting` / `diverging` /
  `plateaued`) that names the val-optimal checkpoint to keep.
- Progress bar + ETA, throughput (tok/s), peak RAM, and gradient-norm health.
- `mlglance demo` synthetic run with `converge` / `overfit` / `diverge` scenarios.
- `--watch`, `--log-y`, `--ascii`, `--no-color`, `--total`, `--title`,
  `--width`/`--height`, `--interval`, and `--version` flags.
