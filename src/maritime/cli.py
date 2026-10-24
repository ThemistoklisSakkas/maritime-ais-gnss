"""Command-line entry point: `python -m maritime <command>`.

Each phase registers one subcommand here as it is implemented.
"""

from __future__ import annotations

import argparse

from maritime.config import load_config


def cmd_info(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    print(f"seed        : {cfg.seed}")
    print(f"study days  : {cfg.start} .. {cfg.end} ({len(cfg.days)} days)")
    print(f"metric CRS  : {cfg.crs_metric}")
    for name, bbox in cfg.aoi.items():
        print(f"aoi[{name}] : {bbox}")
    print(f"data/raw    : {cfg.paths.raw}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="maritime")
    parser.add_argument("--config", default=None, help="path to YAML config")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("info", help="print the resolved configuration").set_defaults(func=cmd_info)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
