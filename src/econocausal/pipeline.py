"""Run the whole EconoCausal pipeline in one command.

Steps, in order:

    download      download Online Retail II from UCI (skipped if already downloaded)
    load          convert the Excel file to parquet (skipped if the parquet file exists)
    build         clean, build the weekly panel, save to SQLite
    baselines     naive OLS baselines (synthetic and real data)
    dml           DML: average, per category, per product (causal forest), EconML check
    robustness    placebo, random common cause and subset checks
    optimize      price recommendations for every product
    serving       small database for the deployed API (deploy/econocausal_serving.db)

Usage:
    python -m econocausal.pipeline                      # everything
    python -m econocausal.pipeline --from build         # start at a later step
    python -m econocausal.pipeline --only dml optimize  # selected steps
    python -m econocausal.pipeline --force-download     # download and convert again

All model steps use fixed random seeds, so a re-run reproduces the same results.
"""

import argparse
import logging
import time
from collections.abc import Callable

from econocausal.log import setup_logging

logger = logging.getLogger(__name__)


def _download(force: bool) -> None:
    from econocausal.data.download import download_dataset

    download_dataset(force=force)


def _load(force: bool) -> None:
    from econocausal.data.load import PARQUET_PATH, build_parquet

    if PARQUET_PATH.exists() and not force:
        print(f"Already converted: {PARQUET_PATH}")
        return
    build_parquet()


def _main_of(module: str) -> Callable[[bool], None]:
    def run(_force: bool) -> None:
        import importlib

        importlib.import_module(module).main()

    return run


STEPS: dict[str, Callable[[bool], None]] = {
    "download": _download,
    "load": _load,
    "build": _main_of("econocausal.data.build"),
    "baselines": _main_of("econocausal.baselines"),
    "dml": _main_of("econocausal.run_dml"),
    "robustness": _main_of("econocausal.robustness"),
    "optimize": _main_of("econocausal.optimize"),
    "serving": _main_of("econocausal.serving"),
}


def select_steps(only: list[str] | None = None, start: str | None = None) -> list[str]:
    """Names of the steps to run, in pipeline order."""
    names = list(STEPS)
    if only:
        unknown = sorted(set(only) - set(names))
        if unknown:
            raise ValueError(f"Unknown steps {unknown}. Choose from {names}.")
        return [n for n in names if n in only]
    if start:
        if start not in STEPS:
            raise ValueError(f"Unknown step '{start}'. Choose from {names}.")
        return names[names.index(start):]
    return names


def run(steps: list[str], force_download: bool = False) -> dict[str, float]:
    """Run the given steps in order and return the seconds each one took."""
    timings = {}
    for i, name in enumerate(steps, start=1):
        print(f"\n=== [{i}/{len(steps)}] {name} ===")
        logger.info("step %d/%d '%s' started", i, len(steps), name)
        start = time.time()
        try:
            STEPS[name](force_download)
        except Exception:
            logger.exception("step '%s' failed after %.0fs", name, time.time() - start)
            raise
        timings[name] = time.time() - start
        logger.info("step '%s' finished in %.0fs", name, timings[name])
    return timings


def main(argv: list[str] | None = None) -> None:
    """Command-line entry point: parse --only / --from / --force-download, run the
    selected steps and print how long each took."""
    parser = argparse.ArgumentParser(description="Run the EconoCausal pipeline.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--only", nargs="+", metavar="STEP", help=f"run only these steps: {', '.join(STEPS)}")
    group.add_argument("--from", dest="start", metavar="STEP", help="start at this step")
    parser.add_argument("--force-download", action="store_true",
                        help="download and convert the raw data again even if it exists")
    args = parser.parse_args(argv)
    setup_logging()

    steps = select_steps(args.only, args.start)
    total = time.time()
    timings = run(steps, args.force_download)

    print("\n=== Done ===")
    for name, seconds in timings.items():
        print(f"{name:<12} {seconds:>6.0f}s")
    print(f"{'total':<12} {time.time() - total:>6.0f}s")


if __name__ == "__main__":
    main()
