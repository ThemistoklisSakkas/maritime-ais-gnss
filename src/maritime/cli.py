"""Command-line entry point: `python -m maritime <command>`.

Each phase registers one subcommand here as it is implemented.
"""

from __future__ import annotations

import argparse
import sys
from datetime import date

from maritime.config import load_config


def cmd_info(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    print(f"seed        : {cfg.seed}")
    print(f"study days  : {cfg.start} .. {cfg.end} ({len(cfg.days)} days)")
    print(f"metric CRS  : {cfg.crs_metric}")
    for name, bbox in cfg.aoi.items():
        print(f"aoi[{name}] : {bbox}")
    print(f"data/raw    : {cfg.paths.raw}")


def cmd_a1(args: argparse.Namespace) -> None:
    from maritime.ais.pipeline import run_a1

    cfg = load_config(args.config)
    days = [date.fromisoformat(d) for d in args.days] if args.days else None
    run_a1(cfg, days=days, force=args.force)


def cmd_a2(args: argparse.Namespace) -> None:
    from maritime.ais.pipeline import run_a2

    run_a2(load_config(args.config))


def cmd_a3(args: argparse.Namespace) -> None:
    from maritime.ais.pipeline import run_a3

    run_a3(load_config(args.config))


def cmd_a4(args: argparse.Namespace) -> None:
    from maritime.ais.pipeline import run_a4

    run_a4(load_config(args.config))


def cmd_a5(args: argparse.Namespace) -> None:
    from maritime.ais.pipeline import run_a5

    run_a5(load_config(args.config))


def cmd_b1(args: argparse.Namespace) -> None:
    from maritime.nav.pipeline import run_b1

    run_b1(load_config(args.config))


def cmd_b2(args: argparse.Namespace) -> None:
    from maritime.nav.pipeline import run_b2

    run_b2(load_config(args.config))


def main(argv: list[str] | None = None) -> None:
    # Windows consoles default to cp1252, which cannot print polars' table borders
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(prog="maritime")
    parser.add_argument("--config", default=None, help="path to YAML config")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("info", help="print the resolved configuration").set_defaults(func=cmd_info)

    a1 = sub.add_parser("a1", help="download, clean and segment AIS data")
    a1.add_argument("--days", nargs="+", help="YYYY-MM-DD; default: all days in config")
    a1.add_argument("--force", action="store_true", help="redo days already processed")
    a1.set_defaults(func=cmd_a1)

    sub.add_parser("a2", help="dark activity: AIS gaps and abrupt manoeuvres") \
       .set_defaults(func=cmd_a2)
    sub.add_parser("a3", help="ship-to-ship encounters (KD-tree pairing)") \
       .set_defaults(func=cmd_a3)
    sub.add_parser("a4", help="GNSS interference indicators and heatmap") \
       .set_defaults(func=cmd_a4)
    sub.add_parser("a5", help="ETA prediction: baseline vs gradient boosting vs crew ETA") \
       .set_defaults(func=cmd_a5)
    sub.add_parser("b1", help="simulate trajectory and sensors; INS drift and Allan check") \
       .set_defaults(func=cmd_b1)
    sub.add_parser("b2", help="INS/GNSS EKF: Monte Carlo accuracy and consistency") \
       .set_defaults(func=cmd_b2)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
