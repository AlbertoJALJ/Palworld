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
from .gamefiles import GameFilesSource
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


def cmd_gamefiles(args: argparse.Namespace) -> int:
    """Build a dataset from extracted game DataTables. The preferred source."""
    source = GameFilesSource(
        root=args.root, locale=args.locale, game_version=args.game_version
    )
    try:
        dataset = source.fetch()
    except IngestError as exc:
        print(f"ingest failed: {exc}", file=sys.stderr)
        return 1

    if source.warnings:
        print(f"{len(source.warnings)} warning(s):", file=sys.stderr)
        for warning in source.warnings[:20]:
            print(f"  {warning}", file=sys.stderr)
        if len(source.warnings) > 20:
            print(f"  ... and {len(source.warnings) - 20} more", file=sys.stderr)
        print(
            "  Warnings naming pals with no shipped display name are unreleased "
            "content and are expected.",
            file=sys.stderr,
        )

    write_dataset(dataset, args.output)
    print(f"Wrote {args.output}")
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


def cmd_crosscheck(args: argparse.Namespace) -> int:
    """Compare a dataset against an independent extraction of the same data.

    Two extractions built by different people from the same game files should
    agree on every breeding rank. Where they do not, one of them is wrong, and
    a route planned on the wrong one is wrong in a way nothing else here would
    catch. Expects a directory holding `pals/*.json` and `breeding.json` in the
    palworld-atlas-data layout.
    """
    try:
        dataset = load_dataset(args.path)
    except DatasetError as exc:
        print(f"invalid: {exc}", file=sys.stderr)
        return 1

    reference_dir = Path(args.reference)
    pal_files = sorted((reference_dir / "pals").glob("*.json"))
    if not pal_files:
        print(f"no pals/*.json under {reference_dir}", file=sys.stderr)
        return 1

    reference: dict[str, dict] = {}
    for file in pal_files:
        row = json.loads(file.read_text(encoding="utf-8"))
        if isinstance(row, dict) and "name" in row:
            reference[row["name"]] = row

    mine = {p.name: p for p in dataset.pals}
    shared = sorted(set(mine) & set(reference))
    only_mine = sorted(set(mine) - set(reference))
    only_reference = sorted(set(reference) - set(mine))

    mismatches = [
        (name, mine[name].combi_rank, reference[name].get("breedingRank"))
        for name in shared
        if mine[name].combi_rank != reference[name].get("breedingRank")
    ]

    print(f"roster: mine={len(mine)} reference={len(reference)} shared={len(shared)}")
    for label, names in (("only in mine", only_mine), ("only in reference", only_reference)):
        if names:
            print(f"  {label} ({len(names)}): " + ", ".join(names[:15]))
    print(f"breeding rank: {len(shared) - len(mismatches)}/{len(shared)} agree")
    for name, ours, theirs in mismatches[:20]:
        print(f"  {name}: ours={ours} reference={theirs}")

    combos_file = reference_dir / "breeding.json"
    if combos_file.exists():
        payload = json.loads(combos_file.read_text(encoding="utf-8"))
        theirs = {
            (*sorted((c["parentAId"].lower(), c["parentBId"].lower())), c["childId"].lower())
            for c in payload.get("uniquePairs", [])
        }
        ours = {(*sorted(c.parents), c.child) for c in dataset.special_combos}
        unsupported = sorted(ours - theirs)
        print(f"combos: mine={len(ours)} reference={len(theirs)} shared={len(ours & theirs)}")
        if unsupported:
            # Combos only we have are the dangerous direction: they would send a
            # player down a route the reference says does not exist.
            print(f"  not in reference ({len(unsupported)}): {unsupported[:10]}")

    failed = bool(mismatches or only_mine or only_reference)
    print("\nMISMATCH" if failed else "\nSources agree.")
    return 1 if failed else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="palworld-ingest", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="write the synthetic demo dataset")
    demo.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    demo.set_defaults(func=cmd_demo)

    gamefiles = sub.add_parser(
        "gamefiles", help="build a dataset from extracted game DataTables (preferred)"
    )
    gamefiles.add_argument(
        "root",
        type=Path,
        help="directory containing Pal/Content/... from an asset extraction",
    )
    gamefiles.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    gamefiles.add_argument("--locale", default="en")
    gamefiles.add_argument("--game-version", default=None)
    gamefiles.set_defaults(func=cmd_gamefiles)

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

    crosscheck = sub.add_parser(
        "crosscheck", help="compare a dataset against an independent extraction"
    )
    crosscheck.add_argument("path", type=Path, nargs="?", default=DEFAULT_OUTPUT)
    crosscheck.add_argument(
        "reference", type=Path, help="directory with pals/*.json and breeding.json"
    )
    crosscheck.set_defaults(func=cmd_crosscheck)

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
