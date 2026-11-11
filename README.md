# Evidence-Market Fault Localization

Small-model repository fault localization. Five specialist agents (trace, graph, spec, static, semantic) write short
notes about candidate functions. A frozen Gemma *probability agent* (`google/gemma-4-E4B-it`) reads any subset of those
notes and outputs a distribution over 20 candidates. The 32-subset value tables let us test whether aggregating
specialist evidence with information-aware weights beats a single Gemma call given the same evidence, and measure what
each evidence source adds.

- **`PLAN.md`** is the source of truth for the experiment design, gates (G0–G4), and rules.
- **`ANALYSIS_PLAN.md`** is the pre-registered statistics plan. The primary test (E1) is Top-1 of P_hedge vs B_full5 on
  the MarketBugs test split (`test_informed`), using a repo-cluster bootstrap.
- **`BLOCKERS.md`** logs decisions, deviations, VERIFY items and dated status.

## Layout

```
configs/    models.yaml (pinned model revisions), repos.yaml (MarketBugs repos + split), prompts/
schemas/    instance / candidates / notes / values JSON Schemas (frozen after G0)
src/gen/    MarketBugs: repo evaluation, mutant generation, strata, trace sanitizing
src/swe/    SWE-bench Lite sampling + gold-function mapping
src/cands/  function index, BM25 / embedding / call-graph retrieval, 20-candidate pools
src/agents/ specialists, probability agent, value tables, pilot driver
src/llm/    Modal server (Gemma HF forward pass for label logits + generation; bge-small embedder)
src/baselines/  B3 Ochiai spectrum-based FL
src/analysis/   scoring, dev-only retrieval checks
data/       marketbugs/{dev,test}/<repo>.jsonl (+ evaluator-only eval_only.jsonl), swebench_lite/
artifacts/  logs, costs.jsonl, eval/gen logs, pilot outputs
```

## Quick start

```bash
make test        # unit tests (schemas, extraction, no-leak, Shapley, parsing)
make validate    # schema-validate every dataset row
.venv/bin/python -m src.gen.eval_repos            # evaluate candidate repos (clones into data/marketbugs/repos/)
.venv/bin/python -m src.gen.build_marketbugs <repo>   # build mutants for one approved repo
.venv/bin/modal deploy -m src.llm.modal_server    # deploy GPU server (billed only while called)
.venv/bin/python -m src.agents.pilot artifacts/pilot_dev30.jsonl --n 30 --max-usd 10.79
```

The cloned repos and their venvs (`data/**/repos/`) are git-ignored. Rebuild them with `src.gen.eval_repos`.

## Progress

### G0: scaffold (done, 2026-11-09/10)
- Schemas v2, validators, label-logit extraction protocol, leakage test, analysis plan committed.
- GPU smoke test passed on Modal A10G: Gemma 4 E4B-it loads, labels `' A'..' T'` are single tokens, 3/3 probes
  correct, and a rerun is bit-identical.
- SWE-bench Lite sample: 100 of 300 test tasks (seed 20261110, proportional by repo). 79 are single-function,
  17 multi-function and 4 file-level.

### MarketBugs dataset (done, 2026-11-10)
- 16 candidate libraries were evaluated for license, pure-Python code, 3× suite timing and determinism. **10 were
  approved:** toolz, sortedcontainers, more-itertools, boltons, cachetools, dateutil, tomli, isodate, intervaltree,
  attrs (one environment-only test deselected).
- Seeded repo split (20261110), fixed before any results: **dev** = cachetools, intervaltree, sortedcontainers;
  **test** = the other 7.
- **330 validated instances**: dev 120 (3×40) and test 210 (7×30). Each is a single-function mutant that fails at
  least one test, deterministically over 3 runs.
- Strata: 200 direct, 41 near, 89 far.
- Traces are sanitized: repo-relative paths, no host paths, no wall-clock footers.

### Pre-pilot checks (dev split only, CPU)
| dev repo | BM25 Top-1 | BM25 Top-5 | gold in 20-pool (lower bound, no embeddings) |
|---|---|---|---|
| cachetools | 0.225 | 0.60 | 0.875 |
| intervaltree | 0.075 | 0.375 | 0.85 |
| sortedcontainers | 0.30 | 0.50 | 0.85 |

### G1 pilot (in progress)
- 30 dev instances (10 per dev repo), full 32-subset tables. Hard budget cap of $10.79 enforced in code.
- Fixed while bringing the pilot up:
  - The embedder image was missing its config, so the container crash-looped.
  - Full-sequence logits ran out of memory on the A10G, so only the last position is now computed.
  - The note parser dropped Gemma's bracketed `RANKED:` format, which lost about 80% of the notes.
  - A stall watchdog was added.
- Measured cost is about $0.04 per instance.

### Next
- Pilot results and the G1 checks: `p_empty` vs random, leakage, order sensitivity, cost extrapolation, note
  spot-check.
- B3 Ochiai on the dev split. G2 freeze: temperature, Hedge η, `configs/FROZEN.txt`.
