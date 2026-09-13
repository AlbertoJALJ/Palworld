# Palworld Data API

Pal data, breeding combinations, breeding routes and passive optimisation,
served over HTTP with generated OpenAPI docs.

The interesting part is not the data store, it is the two engines on top of it:
a route planner that answers *"I have these pals, how do I get to that one?"*,
and a passive optimiser that answers *"what should this pal be carrying, for
base work or for combat?"* — with the odds of actually breeding it.

---

## Data

The API ships with **real data, read out of the game's own DataTables** —
`DT_PalMonsterParameter`, `DT_PalCombiUnique` and `DT_PassiveSkill_Main`, the
tables the game itself reads. 288 pals, 282 passives, 184 unique breeding pairs.
No scraping, no wiki lag behind a patch, no parsing guesswork.

It is **cross-checked against a second, independent extraction**. Two people
extracting the same game files should agree on every breeding rank; where they
do not, one is wrong, and a route planned on the wrong one is wrong in a way
nothing else here would catch. Currently 288/288 ranks agree and no combo exists
in this dataset that the reference does not have. The check is a command, so it
can be re-run against each patch:

```bash
python -m palworld_api.ingest.cli crosscheck data/pals.json <reference-build-dir>
```

### Regenerating it

```bash
python -m palworld_api.ingest.cli gamefiles <extraction-root> --output data/pals.json
```

`<extraction-root>` is a directory holding `Pal/Content/...` from a UE asset
extraction — produced with FModel or repak against your own install, or taken
from one of the community extraction repositories. The adapter reports what it
skipped and why; warnings naming pals with no shipped display name are
unreleased content and are expected.

> **Licensing.** The dataset is factual game data (names, ranks, stats) derived
> from Pocketpair's files. Redistributing it is normal practice across Palworld
> tooling, but it is your call: `data/pals.json` is regenerable from any
> extraction in one command, so it can be deleted from the repo without losing
> anything.

### The other two sources

`data/demo.json` is a synthetic dataset of deliberately invented pals, used for
smoke-testing the API without touching real data. A demo built from real pal
names with placeholder numbers is indistinguishable from real data three screens
into a JSON response, and someone would eventually plan a real breeding run on
numbers that were never measured.

`ingest/paldb.py` scrapes a community wiki. It is a fallback, kept because it
needs no extraction step — but **its selectors have never run against the live
site**, since the environment it was written in blocks outbound access to the
source. Validate it before trusting it:

```bash
python -m palworld_api.ingest.cli inspect https://paldb.cc/en/Anubis
python -m palworld_api.ingest.cli scrape --limit 5
```

Every API response carries a `data_quality` block, so a client can tell
game-file data from wiki or demo data without reading this file.

## Icons

The web UI shows each pal's own icon, not just its name, in every picker and
result list. Same approach as the dataset: read out of the game's own texture
extraction rather than drawn or scraped, matched to a pal by the same tribe id
`gamefiles.py` already uses, so no new naming convention to maintain.

```bash
python -m palworld_api.ingest.cli icons <extraction-root> --dataset data/pals.json
```

`<extraction-root>` is the same kind of directory `gamefiles` reads, this time
looking under `Pal/Content/Pal/Texture/PalIcon/Normal/`. Currently 288/288 pals
have an icon; a pal with none shows a colored initial instead of a broken image
(see `WorkRanking`-style honesty elsewhere in this project — a gap is reported,
never guessed at).

> **Licensing.** Same note as the dataset, same answer: these are Pocketpair's
> own icon textures, extracted rather than drawn. This project is open source
> and non-commercial, which is why the icons are checked in, but that is a
> mitigating factor, not a license grant — it is your call whether to keep
> them. `src/palworld_api/web/icons/*.png` is regenerable from any extraction
> in one command, so removing it costs nothing but re-running `icons`.

## Quick start

```bash
pip install -e ".[dev]"
uvicorn palworld_api.api:app --reload
```

That serves `data/pals.json`, the real dataset, which is already in the repo.
Point `PALWORLD_DATASET` elsewhere to serve a different one.

- Web UI: <http://127.0.0.1:8000/>
- OpenAPI docs: <http://127.0.0.1:8000/docs>

---

## How breeding works here

Three rules, in priority order:

1. **Special combo** — a fixed parent pair with a fixed child. Overrides everything.
2. **Same species** — always breeds true.
3. **The rank formula** — `target = floor((rank_a + rank_b + 1) / 2)`, then the
   breedable child whose `combi_rank` is nearest to `target`.

Two things the game data makes explicit that a wiki tends to bury:

- **`IgnoreCombi`** marks a pal the rank formula must never touch, in either
  direction. Those pals still breed, but only through an explicit unique combo —
  which is exactly how the legendaries end up able to produce nothing but
  themselves. No special-casing needed: rule 1 already runs first.
- **One pair in the entire game depends on the parents' sexes.** Katress + Wixen
  makes Katress Ignis or Wixen Noct depending on which is female. A model with
  one child per pair silently loses one of the two, so `SpecialCombo` carries
  optional genders and the route planner treats both children as reachable.

### Why routes are non-trivial

The formula always lands *between* the two parent ranks. It interpolates; it
never extrapolates. Breeding common pals together can therefore never reach a
rarer one — you need a pal already at that rank, or a special combo. That single
property is the reason a "route to Anubis" is a real problem rather than a
lookup, and it is pinned by a test
(`test_formula_never_leaves_the_parents_rank_range`).

### The search

An edge needs *two* sources at once, so this is a shortest-hyperpath problem
over hyperedges `(a, b) -> c`, not an ordinary graph. Plain Dijkstra does not
apply; Knuth's generalisation does, because both cost functions are monotone and
superior (`f(x, y) >= max(x, y)`). Breeding also does not consume the parents, so
the set of species you own only grows — that monotonicity is what lets each pal
be finalised once.

Two cost models, because they answer different questions:

| Strategy | Cost | Use when |
|---|---|---|
| `generations` (default) | `max(a, b) + 1` | Time is scarce — parent lines are raised in parallel |
| `tree` | `a + b + 1` | Eggs are scarce — counts shared intermediates twice |

---

## Passives

Passives are modelled as additive percentage modifiers on a small set of stats.
A `GoalProfile` is a weight per stat; `base` and `combat` are just two particular
sets of weights, and you can pass your own.

Selection is **exact, not greedy**: with additive scores and an "at most one per
exclusive group, at most N total" constraint, taking the best member of each
group and then the N best of those is provably optimal. A test checks it against
brute force.

Three things the scoring gets right that a hand-written tier list does not:

- **Reducing a loss is a gain.** Sanity- and hunger-loss stats carry negative
  weights, so a `-15% sanity loss` passive scores positively — and much higher
  for a base pal than a fighter.
- **Conditional passives are discounted.** A night-only `+30% attack` scores
  below a flat `+20%` at the default 50% reliability. Tunable.
- **Work passives are worthless on a pal that cannot work.** Work-speed scoring
  scales with the pal's best work suitability, so a combat-only pal never gets
  recommended a work passive.
- **Element buffs only count for pals of that element.** A passive granting
  +30% fire and +30% electric damage is worth exactly nothing on a ground pal.
  Counting it anyway put four useless passives at the top of Anubis's combat
  list, ahead of Legend, until this was fixed.

---

## Passive inheritance

The species route says which pals to breed. It says nothing about the part that
actually costs time: re-rolling eggs until the child keeps the passives you want.

The model: the child's candidate pool is the union of both parents' passives, it
inherits `N` of them drawn from a weight table, and the `N` are drawn uniformly
without replacement. That last step makes an exact answer possible — the chance a
specific set of `k` passives all survive a draw of `N` from a pool of `M` is
`C(M-k, N-k) / C(M, N)`, so the whole thing is a weighted sum of hypergeometric
terms and needs no simulation.

It is checked three ways: against a Monte Carlo simulation, against exhaustive
enumeration, and against the independent identity `P(one passive) = E[N] / M`.

`POST /routes/plan` walks a route and reports, per step, the probability, the
expected number of eggs, and the eggs needed for 90% confidence.

> The weights in `InheritanceConfig` are **community-measured, not read out of
> the game**, and they move between patches. They live in config with their own
> provenance for exactly that reason — override them rather than trusting them.

---

## API

| Endpoint | What it answers |
|---|---|
| `GET /health` | Dataset contents and data quality |
| `GET /pals` | List/filter by element, work, breedability, name |
| `GET /pals/{id}` | Full record |
| `GET /breed?parent_a=&parent_b=` | What one pair produces, by which rule, and any sex requirement |
| `GET /pals/{id}/parents` | Every pair that produces this pal |
| `GET /pals/{id}/children` | What this pal produces with every partner |
| `GET /routes/{target}` | Cheapest route, plus alternatives for the final step |
| `GET /reachable` | Everything reachable from a starting set |
| `GET /goals` | The built-in passive goals and their weights |
| `GET /pals/{id}/passives?goal=` | Optimal loadout for base or combat |
| `GET /work` | The best pals for every base activity (watering, mining, ...) |
| `GET /work/{work}/best` | The best pals for one specific activity |
| `POST /routes/plan` | A route costed out for the passives you want |

```bash
# 19 generations and 75 crosses, from the three starter pals to Anubis.
curl "localhost:8000/routes/Anubis?owned=Lamball&owned=Cattiva&owned=Chikipi"

curl "localhost:8000/pals/Anubis/passives?goal=base"

# The one pair in the game whose child depends on the parents' sexes.
curl "localhost:8000/breed?parent_a=Katress&parent_b=Wixen"

# The same route, costed out in eggs for the passives you want to carry: ~120.
curl -X POST localhost:8000/routes/plan -H 'content-type: application/json' -d '{
  "target": "Anubis",
  "owned": ["Lamball", "Cattiva", "Chikipi"],
  "desired_passives": ["Legend", "Musclehead"],
  "starting_passives": {"Lamball": ["Legend"], "Cattiva": ["Musclehead"]}
}'
```

Display names and internal ids are interchangeable everywhere: Lamball's id is
`sheepball`, and Musclehead's is `noukin`.

---

## Layout

```
src/palworld_api/
  models.py       Canonical schema. Provenance is part of it, not a footnote.
  dataset.py      Loading and the indexes the engines need to be fast
  breeding.py     The three rules, the N×N table, and its inverse
  routes.py       Knuth's generalised Dijkstra over the breeding hypergraph
  passives.py     Scoring and provably-optimal loadout selection
  inheritance.py  Exact inheritance odds, and eggs-per-step
  api.py          FastAPI surface
  web/            The UI the API serves at /
  ingest/
    gamefiles.py  The game's own DataTables. The real source.
    icons.py      Pal icon textures, matched by the same tribe id
    paldb.py      Community-wiki scraper. Fallback, selectors unvalidated.
    demo.py       Synthetic pals, for smoke tests
    cli.py        Build, validate, cross-check, sync icons
```

The dependency runs one way: nothing in `ingest/` is imported by the engine, so
a broken or replaced source cannot affect the breeding maths.

---

## Design notes

**A missing value is never guessed.** A pal scraped without a `combi_rank` keeps
`combi_rank=None` and is excluded from breeding entirely, rather than being given
a plausible default. A wrong rank produces confidently wrong routes, which is
worse than no route.

**Provenance is in the schema.** Every record can say where it came from and how
much to trust it, and the API surfaces that on every response.

**The demo pals are invented on purpose.** A demo built from real pal names with
placeholder numbers is indistinguishable from real data three screens into a JSON
response, and someone would eventually plan a real breeding run on numbers that
were never measured.

---

## Development

```bash
python -m pytest        # 168 tests
python -m ruff check src tests
```

The test dataset is synthetic, with ranks chosen so every expected result can be
worked out by hand. Testing the engine against real game data would conflate a
broken algorithm with a bad scrape, and only the first is the suite's business.
