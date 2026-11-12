# Project log

Dated progress notes for each gate (G0 to G3), moved from the README. Decisions and deviations are in `BLOCKERS.md`.


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

### Pre-freeze corrections + Mode S (2026-11-10)
Fixed a duplicated traceback in the probability-agent prompt, trace-derived retrieval under `no_trace`, character
(not token) truncation, and −∞ credit for retrieval misses. Leakage tests now cover every prompt builder.
Implemented Mode S (sequential market, canonical and reversed orders, evaluator-side rewards) and a condition-aware,
resumable pipeline that writes schema-valid rows plus telemetry sidecars. Details are in `BLOCKERS.md`.

### G1 pilot (done 2026-11-11): 30 dev instances, corrected prompts
| check | result |
|---|---|
| p_empty beats random | ✅ Top-1 0.633 vs 0.047 |
| no leakage | ✅ 46 tests |
| candidate-order sensitivity | ⚠️ 25.7% > 15%, so the registered 2-permutation averaging protocol is adopted |
| cost fits budget | ⚠️ ≈$49 at full scope vs $45; ≈$34 with cut order #4/#5 (decision pending) |
| notes not copied from trace | ✅ mean overlap 9%, spot-check passed |

Exploratory dev Top-1 (single permutation, n=30): p_empty .63, p_full .73, **B_full5 .77**, P_hedge .73,
Mode S .60–.67. Modal billed $3.35 so far.

### G2 results (frozen 2026-11-12): 120 dev instances, test_informed, 2-permutation averaging
Frozen contract `configs/FROZEN.txt` (code `1ac82a0`), parameters `artifacts/frozen_params.json`, full dev report
`artifacts/g2_dev_report.json`. **Dev numbers, fit and scored on the same 120 instances: exploratory, not findings.**
No test-split inference has been run.

| method | Top-1 | Top-3 | MRR | NLL (tempered, gold in pool) | ECE |
|---|---|---|---|---|---|
| random in pool | .043 | | | | |
| p_empty (no notes) | .617 | .783 | .703 | .852 | .261 |
| p_full (5 notes) | .675 | .792 | .738 | .723 | .267 |
| **B_full5** (5 note orders) | **.692** | .792 | **.748** | .697 | .243 |
| best single agent (trace / spec) | .667 | .800 / .808 | .732 / .735 | .768 / .783 | .252 / .274 |
| P_vote | .692 | .775 | .745 | 1.071 | .183 |
| P_avg | .683 | .792 | .741 | .727 | **.162** |
| P_hedge_eq | .683 | .792 | .745 | .663 | .221 |
| P_hedge (frozen η, weights) | .683 | .792 | .743 | **.660** | .218 |
| Mode S, mean of 2 orders | .675 | .792 | .739 | .749 | .222 |

- **Retrieval sets the ceiling:** gold is in the 20-candidate pool for 102/120 (85%); the 18 misses cap every method.
- **The model is strong without notes; notes add a little.** p_empty .62 → p_full .68 → B_full5 .69 (73% → 81% when
  gold is in the pool). Methods from .66 to .69 differ by 1–3 instances (1 instance = .008): a tie at n=120.
- **Pooling mainly improves calibration, not accuracy.** All methods are overconfident (fitted temperatures ≈ 1.8–2.2);
  pooled log-loss .66 vs .72 for p_full at similar Top-1.
- **Learned weights add almost nothing.** Hedge weights: trace .38, semantic .24, spec .15, static .12, graph .11, but
  P_hedge ≈ equal-weight pool (NLL .660 vs .663). Frozen η = 0.1 is the **lowest value in the pre-registered grid**
  (CV NLL .89 / .96 / 1.07 / 1.11 for η = 0.1 / 0.3 / 1 / 3), so the optimum may be lower (closer to equal weights).
- **2-permutation averaging was needed:** candidate-order sensitivity rose to 33% (G1: 25.7%; threshold 15%). Mode S's
  final pick depends on agent order for 18% of instances.
- **Credit attribution is small and order-dependent.** Shapley (value of an evidence source given the fixed probability
  agent): semantic +.08, trace +.02, static +.01, spec −.06, graph −.07. Mode S rewards change sign with order (graph
  +.01 canonical vs +.16 reversed; semantic +.20 vs −.09).
- **Notes are not copied from the trace:** 100% parsed; mean longest-common-substring overlap 9% (max 42%, up from G1's
  24%; worth a spot-check).
- Per repo Top-1 (B_full5): cachetools .65, intervaltree .73, sortedcontainers .70.

**Protocol amendment before the freeze:** issue text over 4,096 tokens keeps its first 1,024 and last 3,072 tokens.
It changes exactly 2 dev instances (runaway doctest output: 29k and 4.8M tokens); all other prompts are
byte-identical. Details, OOM handling and run provenance (all rows on Modal A10G) are in `BLOCKERS.md`.
Cost of G2 collection ≈ $8.5 conservative, covered by free Modal credit.

### G3 held-out result (done 2026-11-12): 210 test bugs, 7 unseen repos, frozen protocol
| | P_hedge (learned supervised pool) | B_full5 (matched full-evidence baseline) |
|---|---|---|
| Top-1, all 210 | **134 (63.8%)** | **136 (64.8%)** |
| Top-1, gold in pool (172) | 77.9% | 79.1% |

**Preregistered paired difference P_hedge − B_full5: −0.95 points, 95% repository-cluster bootstrap CI [−4.3, +2.9]
(10,000 replicates, seed 20261110): inconclusive.** The methods have the same correctness on 202/210 bugs (W/L/T 3/5/202) and select the same function on 193/210. Pool recall
172/210 (81.9%). Exploratory: notes vs no notes +2.4 points, CI [−3.3, +8.1]; the dev calibration advantage of pooling
reversed on test. Full results: `artifacts/validation/heldout_v1/RESULTS.md`; paper: `paper/MARKETBUGS_PAPER.md`.

### Next
- B3 Ochiai on the dev/test splits; spot-check the highest-overlap notes.
