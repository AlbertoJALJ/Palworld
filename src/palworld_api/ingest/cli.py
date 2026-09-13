"""Command line entry point for building and checking datasets.

    python -m palworld_api.ingest.cli demo                  # write the demo dataset
    python -m palworld_api.ingest.cli scrape --limit 5      # try 5 pages first
    python -m palworld_api.ingest.cli inspect <url>         # see what the parser sees
    python -m palworld_api.ingest.cli validate <file.json>  # check a dataset
    python -m palworld_api.ingest.cli stats <file.json>     # coverage report

`scrape --limit 5` and `inspect` exist because the selectors in the scraper
have never run against the live site. Use them before a full run.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..breeding import BreedingEngine
from ..dataset import DatasetError, PalIndex, load_dataset
from ..models import Dataset
from .base import IngestError
from .demo import build_demo_dataset
from .paldb import PaldbSource

DEFAULT_OUTPUT = Path("data/pals.json")
DEFAULT_CACHE = Path("data/cache")


def write_dataset(dataset: Dataset, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dataset.model_dump(mode="json", exclude_none=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _report(dataset: Dataset) -> None:
    index = PalIndex(dataset)
    engine = BreedingEngine(index)
    ranked = [p for p in dataset.pals if p.combi_rank is not None]
    print(f"  pals:            {len(dataset.pals)}")
    print(f"  with combi_rank: {len(ranked)}")
    print(f"  passives:        {len(dataset.passives)}")
    print(f"  special combos:  {len(dataset.special_combos)}")
    print(f"  breedable pairs: {len(engine.table)}")
    unreachable = engine.unreachable
    print(f"  unreachable:     {len(unreachable)}")
    if unreachable:
        preview = ", ".join(unreachable[:10])
        suffix = " ..." if len(unreachable) > 10 else ""
        print(f"      {preview}{suffix}")
    if not dataset.is_verified:
        print("  NOTE: dataset is not fully verified; some values carry seed provenance.")


def cmd_demo(args: argparse.Namespace) -> int:
    dataset = build_demo_dataset()
    write_dataset(dataset, args.output)
    print(f"Wrote demo dataset to {args.output}")
    print("  The pals in it are invented. Run `scrape` for real data.")
    _report(dataset)
    return 0


def cmd_scrape(args: argparse.Namespace) -> int:
    source = PaldbSource(
        base_url=args.base_url,
        cache_dir=args.cache_dir,
        game_version=args.game_version,
        min_delay=args.delay,
        limit=args.limit,
        index_path=args.index_path,
    )
    try:
        dataset = source.fetch()
    except IngestError as exc:
        print(f"scrape failed: {exc}", file=sys.stderr)
        return 1

    if source.warnings:
        print(f"{len(source.warnings)} warning(s):", file=sys.stderr)
        for warning in source.warnings[:40]:
            print(f"  {warning}", file=sys.stderr)
        if len(source.warnings) > 40:
            print(f"  ... and {len(source.warnings) - 40} more", file=sys.stderr)

    if args.limit is not None:
        print("\nThis was a limited run; review the values above before trusting them.")
    write_dataset(dataset, args.output)
    print(f"Wrote {args.output}")
    _report(dataset)
    return 0


def cmd_inspect(args: argparse.Namespace) -> int:
    """Dump the label/value pairs the parser finds on one page."""
    from .http import PoliteClient

    source = PaldbSource(base_url=args.base_url, cache_dir=args.cache_dir)
    try:
        with PoliteClient(cache_dir=args.cache_dir) as client:
            html = client.get(args.url)
    except IngestError as exc:
        print(f"fetch failed: {exc}", file=sys.stderr)
        return 1

    from selectolax.parser import HTMLParser

    pairs = source._label_value_pairs(HTMLParser(html))
    print(f"{len(pairs)} label/value pairs found on {args.url}:\n")
    for label, values in sorted(pairs.items()):
        print(f"  {label:<28} {values[0][:70]}")

    print("\nParsed result:")
    try:
        parsed = source.parse_pal(html, args.url, "inspect")
    except IngestError as exc:
        print(f"  parse failed: {exc}")
        return 1
    print(json.dumps(parsed.pal.model_dump(mode="json", exclude_none=True), indent=2))
    if parsed.missing:
        print(f"\n  missing: {', '.join(parsed.missing)}")
        print("  Add the site's label for these to the *_LABELS tuples in paldb.py.")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    try:
        dataset = load_dataset(args.path)
    except DatasetError as exc:
        print(f"invalid: {exc}", file=sys.stderr)
        return 1
    print(f"{args.path} is valid.")
    _report(dataset)
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    try:
        dataset = load_dataset(args.path)
    except DatasetError as exc:
        print(f"invalid: {exc}", file=sys.stderr)
        return 1

    missing_rank = [p.id for p in dataset.pals if p.combi_rank is None]
    missing_work = [p.id for p in dataset.pals if not p.work]
    unverified = [
        p.id for p in dataset.pals if p.provenance is None or not p.provenance.verified
    ]
    print(f"source: {dataset.source}  game_version: {dataset.game_version}")
    _report(dataset)
    for label, items in (
        ("missing combi_rank", missing_rank),
        ("missing work data", missing_work),
        ("unverified provenance", unverified),
    ):
        print(f"\n{label}: {len(items)}")
        if items:
            print("  " + ", ".join(items[:20]) + (" ..." if len(items) > 20 else ""))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="palworld-ingest", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="write the synthetic demo dataset")
    demo.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    demo.set_defaults(func=cmd_demo)

    scrape = sub.add_parser("scrape", help="build a dataset from a community wiki")
    scrape.add_argument("--base-url", default="https://paldb.cc")
    scrape.add_argument("--index-path", default="/en/Pals")
    scrape.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    scrape.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    scrape.add_argument("--game-version", default=None)
    scrape.add_argument(
        "--delay", type=float, default=1.5, help="minimum seconds between requests"
    )
    scrape.add_argument(
        "--limit", type=int, default=None, help="only fetch the first N pals"
    )
    scrape.set_defaults(func=cmd_scrape)

    inspect = sub.add_parser("inspect", help="show what the parser sees on one page")
    inspect.add_argument("url")
    inspect.add_argument("--base-url", default="https://paldb.cc")
    inspect.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    inspect.set_defaults(func=cmd_inspect)

    validate = sub.add_parser("validate", help="check a dataset file")
    validate.add_argument("path", type=Path, nargs="?", default=DEFAULT_OUTPUT)
    validate.set_defaults(func=cmd_validate)

    stats = sub.add_parser("stats", help="coverage report for a dataset file")
    stats.add_argument("path", type=Path, nargs="?", default=DEFAULT_OUTPUT)
    stats.set_defaults(func=cmd_stats)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
