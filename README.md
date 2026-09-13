# Palworld Data API

Pal data, breeding combinations, breeding routes and passive optimisation,
served over HTTP with generated OpenAPI docs.

The interesting part is not the data store, it is the two engines on top of it:
a route planner that answers *"I have these pals, how do I get to that one?"*,
and a passive optimiser that answers *"what should this pal be carrying, for
base work or for combat?"* — with the odds of actually breeding it.

---

## Status: the engine is done, the data is not

Everything in `src/palworld_api/` works and is tested. What ships with it is a
**demo dataset of invented pals** (`data/demo.json`), not real game data.

The scraper that produces real data is written and its parsing logic is tested,
but **its selectors have never run against the live site** — the environment it
was written in blocks outbound access to the source. So:

- The engine, API and web UI are verified end to end. 108 tests pass.
- The scraper's HTML parsing is verified against three markup layouts.
- Whether those layouts match the real site is **unknown**. Validate it before
  trusting a full run:

```bash
python -m palworld_api.ingest.cli inspect https://paldb.cc/en/Anubis
python -m palworld_api.ingest.cli scrape --limit 5
```

`inspect` prints every label/value pair the parser finds on one page plus the
`Pal` it built. If a field is missing, add the site's label to the relevant
`*_LABELS` tuple in `ingest/paldb.py` — that is the intended fix, and it needs
no code changes.

Also check the source's terms of use before scraping it. The client honours
`robots.txt`, rate limits, and caches everything, but permission is a separate
question from politeness.

Every API response carries a `data_quality` block, so a client can tell demo
data from verified data without reading this file.

---

## Quick start

```bash
pip install -e ".[dev]"

python -m palworld_api.ingest.cli demo --output data/demo.json
PALWORLD_DATASET=data/demo.json uvicorn palworld_api.api:app --reload
```

- Web UI: <http://127.0.0.1:8000/>
- OpenAPI docs: <http://127.0.0.1:8000/docs>

---

## How breeding works here

Three rules, in priority order:

1. **Special combo** — a fixed parent pair with a fixed child. Overrides everything.
2. **Same species** — always breeds true.
3. **The rank formula** — `target = floor((rank_a + rank_b + 1) / 2)`, then the
   breedable child whose `combi_rank` is nearest to `target`.

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
| `GET /breed?parent_a=&parent_b=` | What one pair produces, and by which rule |
| `GET /pals/{id}/parents` | Every pair that produces this pal |
| `GET /pals/{id}/children` | What this pal produces with every partner |
| `GET /routes/{target}` | Cheapest route, plus alternatives for the final step |
| `GET /reachable` | Everything reachable from a starting set |
| `GET /goals` | The built-in passive goals and their weights |
| `GET /pals/{id}/passives?goal=` | Optimal loadout for base or combat |
| `POST /routes/plan` | A route costed out for the passives you want |

```bash
curl "localhost:8000/routes/demo_sovereign"
curl "localhost:8000/pals/demo_forge/passives?goal=base"
curl -X POST localhost:8000/routes/plan -H 'content-type: application/json' -d '{
  "target": "demo_sovereign",
  "desired_passives": ["demo_legend", "demo_ferocious"],
  "starting_passives": {"demo_titan": ["demo_legend"], "demo_shade": ["demo_ferocious"]}
}'
```

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
  ingest/         Everything that talks to the outside world
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
python -m pytest        # 108 tests
python -m ruff check src tests
```

The test dataset is synthetic, with ranks chosen so every expected result can be
worked out by hand. Testing the engine against real game data would conflate a
broken algorithm with a bad scrape, and only the first is the suite's business.
