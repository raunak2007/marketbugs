# BLOCKERS.md

## Human decisions (PLAN 15)
_2026-11-10_
1. MarketBugs repo list: **DECIDED 2026-11-10** — toolz, sortedcontainers, more-itertools, boltons, cachetools, dateutil, tomli, isodate, intervaltree, attrs (see `configs/repos.yaml`). Split seed 20261110 → dev: cachetools, intervaltree, sortedcontainers; test: other 7.
2. Gemma 4 variant: **DECIDED** `google/gemma-4-E4B-it@4ecc2ba6` (verified to exist on HF, not gated). Alternative `google/gemma-4-E2B-it@4adf1574`. Larger: 12B-it, 26B-A4B-it, 31B-it.
3. Kaggle Data tab: **DECIDED: none / skip.** Graph agent uses an `ast` call graph; Semantic agent uses an open embedding model.
4. E1 primary test: **CONFIRMED by human 2026-11-10** — Top-1 of P_hedge vs B_full5, MarketBugs test, `test_informed`, repo-cluster bootstrap.
5. Thresholds and caps: kept as in PLAN.

## Verified
- Gemma 4 E4B-it tokenizer: ' A'..' T' and 'A'..'T' are each one token; 'Answer:' + ' C' tokenizes as ['Answer', ':', '▁C']. Using the space prefix.

## VERIFY (unverified)
- Gemma 4 E4B-it loads with `AutoModelForCausalLM` in transformers>=5 on Modal (may need the image-text-to-text class).
- A10G 24GB is enough for E4B bf16 at batch 8.
- Code budget is chars (1200), not tokens; switch to tokens before G2.
- SWE-bench Lite dataset id, split sizes.

## Deviations
- [GPT] tasks (mutant generator, BM25) have not started in this repo; Claude Code builds minimal versions if needed (PLAN 0.10).

## Session scope
- 2026-11-10: human asked for the work through the G1 pilot only, then a stop for review.

## Status 2026-11-10 (session 1)
- G0 GPU smoke PASSED: gemma-4-E4B-it on Modal A10G loads as Gemma4ForConditionalGeneration (transformers 5.22.2, torch 2.8.0+cu128); labels ' A'..' T' resolve as single tokens with the real chat template; token_mapping_hash 60e8077ec8db6a85; 3/3 probes top-1 correct, sums 1.0, rerun max diff 0.
- SWE-bench Lite verified: princeton-nlp/SWE-bench_Lite@6ec7bb89, test=300, dev=23. Sample of 100 (seed 20261110, proportional by repo): 79 single-function, 17 multi-function, 4 no-function (file-level bucket). 0 parse or checkout failures.
- **BLOCKER: running external repo test suites was denied by the agent's permission classifier.** MarketBugs generation (5.1), B3 Ochiai and E5 need it. Fallback (PLAN 12): the human runs `python -m src.gen.eval_repos`, then `python -m src.gen.build_marketbugs <repo>` locally, or grants permission.
- Deviation (VERIFY): mutant validity reruns (runs 2–3) re-run only the failing tests, not the full suite.
- VERIFY: vLLM cross-check not needed (an HF forward pass is used). Embedder BAAI/bge-small-en-v1.5 revision not pinned yet.
- Low disk: 5.5 GB free on the user's volume.

## Status 2026-11-10 (session 2)
- Permission granted: `Bash(.venv/bin/python -m src.gen.*)` (project `.claude/settings.local.json`); blocker above resolved for generator steps.
- Fixed `eval_repos` determinism check (compared wall-clock times → always False). Re-scored: 9 repos clean and deterministic 3/3; attrs clean after exclusion below.
- Exclusion: attrs `tests/test_packaging.py::TestLegacyMetadataHack::test_version_info` (2 params) deselected — shallow clone has no tags, version `0.1.dev1` unparseable (environment artifact). Applied via `pytest_args` in repos.yaml.
- Rejected repos: funcy, pyparsing, humanize (collection error); bidict (pytest exit 4; MPL-2.0); parse (exit 4); glom (1 env failure, missing PyYAML; backup).
- Traces are sanitized: repo-relative paths, stdlib → `<python>/`, pytest footer/FAILED lines removed (no host paths/usernames in instances). Strata computed on the sanitized trace.
- `make validate` now globs `data/marketbugs/*/*.jsonl` (excluding `eval_only.jsonl`).
- BUG (strata): builder computed strata on a test-excluded index, so assertion-only traces (only the test function in-frame) could never be `near`. PLAN 5.1 counts any traceback-frame function. Fix: `src.gen.restratify` (test-inclusive index); dry run on first 160 instances moves 17 far→near (totals direct 89 / near 19 / far 52). Apply with `--write` after the build finishes; audit in `artifacts/restratify_audit.json`. Builder's own `stratum()` to be patched the same way afterwards.
- Dev-only pre-pilot check `src.analysis.dev_retrieval_check` (CPU, no Modal). cachetools: B1 BM25 end-to-end Top-1 0.225 / Top-5 0.60; BM25+graph k=20 pool recall lower bound 0.875. No outcomes computed on test.
- NOT DONE: B3 Ochiai (needs per-test coverage runs of external suites under `src.baselines`, outside the granted `src.gen.*` rule) — needs a permission decision.
- 2026-11-10 18:23 MarketBugs BUILT: 330 instances (dev 3×40 = 120; test 7×30 = 210), all caps hit, `make validate` 0 invalid. Restratify applied (`--write`): direct 200 / near 41 / far 89; builder patched to the same rule.
- Text cleanup applied to all 330 rows (`src.gen.resanitize`): removed pytest timing footers ("1 failed in 0.10s", wall-clock noise in model input) and replaced the host home dir with `/home/user` (1 boltons expanduser test printed it). Idempotent; gold/strata unchanged.
- Dev pre-pilot (BM25+graph, no embeddings): B1@1 cachetools .225 / intervaltree .075 / sortedcontainers .30; pool recall LB .875/.85/.85.
- Weak suite note: tomli 691/727 mutants were not detected (17 tests); its 30 instances are valid but skew `far` (17/30).

## Pre-freeze methodological corrections (2026-11-10, session 2; spec "Targeted Methodological Corrections and Mode S Priority")
- **Duplicate traceback (fixed):** `probability_agent_prompt` appended `view["trace"]` to `issue_text`, which already contains the trace under `test_informed`. Now `issue_text` only. Specialist `_issue` helper had the same trap (unused path); removed. Test: trace appears exactly once; absent under `no_trace`.
- **no_trace retrieval boundary (fixed):** pools were always built from the `test_informed` text + trace frames. `src.agents.pipeline.build_candidates` now derives the query and frames from `inference_view(inst, ., cond)`, so `no_trace` pools/evidence/semantic scores see no trace (dry run: no_trace pools have no graph-from-frames source). Pool order seed now includes the condition.
- **Token-aware truncation (fixed):** candidate code truncated to `prompting.code_token_budget: 300` tokens of the pinned Gemma tokenizer (`src.llm.tokens`), replacing 1200 chars. 300 chosen to match the old budget on pilot pools (mean kept 126.9 vs 126.3 tokens) → cost-neutral. No silent char fallback.
- **Gold-not-in-pool (fixed):** `src.analysis.credit` returns None (excluded, counted) for Shapley/pivotal/marginal credit and Mode S rewards when gold is a retrieval miss; unconditional localization metrics still count misses as misses.
- **Leakage test extended** to specialist, forecaster and market prompt builders; plus a gold-invariance test (changing gold/mutation fields leaves every market prompt byte-identical) and a no-outcome-vocabulary check.
- **Frozen schemas respected (deviation from previous pilot code):** note.v2/value_row.v2 have `additionalProperties: false`; the old pilot wrote extra fields (template_hash, input_tokens, latency_ms, raw_text) → invalid rows. Now main rows carry schema fields only; tokens/latency/cost/call kind per call go to `telemetry.jsonl`, raw generations to `raw_notes.jsonl`, K=10 control to `k10.jsonl`. Condition is encoded in the run directory, not a row field. **Schema unchanged.** A `schema_v2.1` adding `condition` and telemetry fields would be cleaner — needs a human decision.
- **Mode S implemented** (`src.market.sequential`, prompt `configs/prompts/market.txt`) per PLAN 6: initial market = p_empty; each agent sees issue + candidates + current top-5 market probabilities + its own note (its permitted evidence, as in PLAN 6); Gemma label extraction gives the updated market; canonical and reversed orders; rows `mode: "S"`, `subset_mask` = agents so far, `note_order` = order so far. Rewards r_i computed evaluator-side only (`src.analysis.credit.market_rewards`); telescoping tested. Not incentive-compatible, not trained.
- **B_full5** = mean of probabilities over the 5 cyclic note-order rotations of the full call (ANALYSIS_PLAN wording "mean"). Rotation 0 is the canonical p_full row.
- **Superseded data:** the 20 pilot instances run before these fixes (cachetools 10, intervaltree 10; ~$1.22 conservative, logged in costs.jsonl) used the duplicated-trace prompt and 1200-char truncation. Kept in `artifacts/pilot_*` for cost/latency reference only; **not used for any G1/G2 decision.**
- Pilot driver rewritten (`src.agents.pilot`): per-instance append + resume, refuses non-dev instances before G2, both conditions, budget guard + stall watchdog kept. Evaluator `src.analysis.evaluate_dev` (dev only).
