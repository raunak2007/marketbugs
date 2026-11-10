# PLAN.md: Evidence-Market Fault Localization (experiment phase)

Read this whole file before writing code. It is the single source of truth for the experiment phase. Paper writing is handled elsewhere and is out of scope.

---

## 0. Operating rules for the coding agent

1. **Schemas are frozen after Gate G0.** Do not change a schema without a human decision. Add fields only via a versioned `schema_v2.1` bump and migration script.
2. **Prompts, pools, scoring and the primary test are frozen at Gate G2.** After G2, only bug fixes that do not change outputs are allowed. Record the commit hash in `configs/FROZEN.txt`.
3. **No gold leakage, ever.** Inference code must never read `gold_*`, `mutation_*`, the SWE-bench gold patch, test patch, or `FAIL_TO_PASS`. Enforce this with a unit test that scans prompt builders (section 4.4).
4. **Everything cacheable is cached** under `artifacts/` with a content hash. Every row records the prompt hash, model revision, seed, tokens, latency and cost.
5. **Log every dollar.** Append each GPU job to `artifacts/costs.jsonl`. Stop and ask the human if cumulative Modal spend passes $45 before G3.
6. **Commit small and often.** One branch per workstream, merge only when tests pass.
7. **If blocked for more than 45 minutes**, take the fallback listed for that item, write the blocker to `BLOCKERS.md`, and continue.
8. **Mark anything you could not verify** with `VERIFY:` in code comments and in `BLOCKERS.md`. Do not invent library behavior, model IDs or dataset sizes.
9. **No new experiments after Wed 23:59 ET.** Thursday is for re-running failures only.
10. Owners: tasks tagged `[CC]` are Claude Code. Tasks tagged `[GPT]` are a second coding assistant working on a separate branch. If a `[GPT]` task is not done when you need it, build a minimal version yourself and note it.

---

## 1. Scientific scope

**Question.** For small-model repository fault localization, does aggregating specialist evidence with information-aware weights beat a single Gemma call given the same evidence, and what does each evidence source add?

**Three modes, never conflated**

| Mode | What it is | Role |
|---|---|---|
| **C: evidence coalition** | 5 specialists write notes. One frozen Gemma *probability agent* reads any subset of notes and outputs a distribution over candidates. Produces the 32-subset value table. | **Primary** |
| **F: forecaster pooling** | Each specialist directly outputs a distribution from its own view. Pooled by vote / average / Hedge / stacking. | Secondary |
| **S: sequential market** | Agents move a shared distribution one at a time, seeing current market values. Reward = change in log score. | Exploratory |

**Naming discipline.** Hedge with dev-fit weights is a *learned supervised pool*, not a market mechanism. Shapley/marginal/pivotal credit measures *the value of an evidence source given a fixed probability agent*, not causal bug-finding ability. Use these exact labels in code, tables and logs.

**Scoring.** Outcome is the gold set G (single function for mutants). Log score `s(p) = log(sum of p over G)`. Value `v(S) = s(p_S) - s(p_empty)`. Note `p_empty` is not evidence-free (issue + candidate code + retrieval-shaped pool).

---

## 2. Repository layout

```
repo/
  PLAN.md  BLOCKERS.md  ANALYSIS_PLAN.md  LICENSE (Apache-2.0)
  configs/        # yaml: models, pools, prompts hashes, FROZEN.txt
  schemas/        # JSON Schema files + validators
  data/
    marketbugs/{dev,test}/instances.jsonl
    swebench_lite/instances.jsonl
  artifacts/      # cached outputs, keyed by hash; costs.jsonl
  src/
    gen/          # mutant generator, env setup
    swe/          # SWE-bench Lite parser
    cands/        # call graph, BM25, embeddings, pool builder
    llm/          # model server wrapper, probability extraction
    agents/       # specialists, probability agent, forecaster, market
    market/       # pooling, Hedge, stacking, Shapley, sequential market
    baselines/    # random, BM25, embedding, Ochiai
    repair/       # single-function repair harness
    analysis/     # metrics, cluster bootstrap, tables, figures
  tests/
  notebooks/reproduce.ipynb   # rebuilds all headline tables from artifacts/
  Makefile        # make data | pools | notes | values | baselines | analyze | reproduce
```

---

## 3. Schemas (v2, JSONL)

```jsonc
// instance
{
  "id": "mb-toolz-0042", "source": "marketbugs|swebench_lite",
  "repo": "toolz", "repo_license": "BSD-3-Clause", "base_commit": "...",
  "split": "dev|test", "issue_text": "...", "trace": "..." ,   // trace null for issue-only
  "conditions": {"test_informed": "...", "no_trace": "..."},   // marketbugs only
  "failing_tests": ["tests/test_x.py::test_y"],                 // marketbugs only; null for swebench inference input
  "gold_functions": [],          // EVALUATOR-ONLY
  "mutation_operator": "cmp_flip", "mutation_span": [l0,c0,l1,c1],   // EVALUATOR-ONLY
  "stratum": "direct|near|far|na", "generation_seed": 123,
  "validation_status": "ok", "test_environment": "py3.11 + lockfile hash"
}
// candidates
{
  "id": "...", "pool_size": 20, "gold_in_pool": true,   // gold_in_pool EVALUATOR-ONLY
  "pool": [{"cid": "A", "path": "...", "qualname": "...", "code": "...",
            "retrieval_source": "bm25|emb|graph", "retrieval_score": 0.0}],
  "candidate_order": [..], "order_seed": 7
}
// note
{
  "id": "...", "agent": "trace|graph|spec|static|semantic", "copy_idx": 0,
  "text": "...", "ranked_suspects": ["C","F","A"],
  "prompt_hash": "...", "model_revision": "...", "temperature": 0.0, "seed": 0,
  "input_tokens": 0, "output_tokens": 0, "latency_ms": 0
}
// value_row
{
  "id": "...", "mode": "C|F|S", "subset_mask": 0..31, "order_perm": 0,
  "note_order": [..], "raw_candidate_logits": [20 floats],
  "probs": [20 floats], "gold_logscore": -1.2, "top1": "C",
  "token_mapping_hash": "...", "prompt_hash": "...", "model_revision": "...",
  "inference_cost_usd": 0.0
}
```

Write JSON Schemas and a `validate` CLI. `make data` must fail on any invalid row.

---

## 4. Core protocols

### 4.1 Probability extraction (Gemma) [CC]
- Label candidates `A` to `T`. Verify with the tokenizer that each label is exactly one token with the same prefix convention (leading space or not). If not, use numerals or another single-token labeling.
- The prompt must end where the next token is the label (for example `Answer:`), after applying the model's chat template if one is used.
- Take logits at the final position, **gather the 20 label-token IDs, softmax over those 20 only.** Never rely on a top-k logprob list. Prefer an HF forward pass; if using vLLM per-token-ID probabilities, cross-check against HF on 10 prompts (agreement within 1e-2 on probabilities). `VERIFY:` supported API in the pinned vLLM version.
- Unit tests (`tests/test_extraction.py`): probabilities sum to 1, finite, repeatable across two runs, label to candidate mapping round-trips, no zero-filling from missing entries, `token_mapping_hash` stable.
- Fallback if Gemma 4 is unsupported in the pinned stack: use the smallest supported Gemma and record the change in `BLOCKERS.md`.

### 4.2 Candidate pool [CC for pipeline, GPT for BM25/embedding retrievers]
- K = 20 functions per instance. Union of BM25 over issue+trace, embedding retrieval, and static call-graph neighbors of traceback frames. Interleave by rank, deduplicate, truncate to 20.
- **Never insert the gold function into the pool.** If gold is missing, it is a retrieval miss.
- Candidate order: seeded shuffle that does not encode retrieval rank. Save `order_seed`.
- Code shown per candidate is truncated to a fixed token budget (set in config; same for every method).
- All methods use the *same* pool. Report **retrieval** (gold in pool) separately from **ranking** (gold ranked first given in pool).
- Check the Kaggle Data tab: if provided code graphs or embeddings are usable, use them for the Graph and Semantic agents and as a retrieval source. `VERIFY:` format. Otherwise build with `ast` and an open embedding model.

### 4.3 Order and size controls (dev only, before G2)
- On 30 to 50 dev instances, run 3 candidate-order permutations. If more than 15% of top-1 predictions change across permutations, the final protocol averages log-probs over 2 fixed permutations for every method (record the decision). The 15% value is a chosen threshold, written into `ANALYSIS_PLAN.md`.
- On the same sample, compare K=10 vs K=20. Report only; the final K stays 20.

### 4.4 Leakage test [CC]
`tests/test_no_leak.py`: build prompts for 50 instances; assert none contain strings from `gold_functions`, `mutation_*`, the SWE-bench gold patch, `test_patch`, or `FAIL_TO_PASS`, except where the same string is legitimately present in the issue or traceback (allow-list by exact span from the permitted input fields). Run in CI on every commit.

---

## 5. Data construction

### 5.1 MarketBugs mutant generator [GPT, CC reviews]
- Choose repos: pure Python, permissive license, deps installable, full suite under about 2 minutes, deterministic. Candidates to evaluate (`VERIFY:` license and speed): toolz, sortedcontainers, more-itertools, boltons, and similar small libraries. Record license in each instance.
- **Minimal source-span edits** (use `ast` positions; never reformat the file).
- Operators: comparison flip, arithmetic/boolean operator swap, off-by-one constant, argument swap, condition removal, wrong-variable substitution, statement deletion, return-value change.
- Validity filters: original suite passes; mutant fails at least 1 test; failure reproduces over 3 runs; exactly one function changed; gold function has at least 3 statements (chosen to exclude trivial one-liners); dedupe by (function, operator, span).
- Issue text (templated from test output):
  - `test_informed`: failing test name, assertion message, traceback.
  - `no_trace`: failing test name and assertion message only.
- Strata (computed, reported as observed, not balanced):
  - `direct`: gold qualname appears in traceback frames.
  - `near`: gold within one static call-hop of a traceback-frame function.
  - `far`: otherwise.
- Sizes (minimum plan): 10 repos, split by repo: 3 dev (cap 40 per repo, at most 120) and 7 test (cap 30 per repo, at most 210). Stretch: 14 repos (4 dev, 10 test, at most 300 test). No train split (the observer LoRA is cut).
- Evaluator-only metadata (patch, span, operator) lives in separate fields never read by inference.

### 5.2 SWE-bench Lite localization set [CC]
- Source: `princeton-nlp/SWE-bench_Lite` (`VERIFY:` dataset id and split sizes, including the dev count).
- Sample 100 test tasks with a fixed seed, stratified by repo, chosen *before any results*. Record the seed. Use dev tasks for pilot sanity only.
- Check out each repo at `base_commit`. Map gold-patch hunks to enclosing functions using pre-patch line ranges.
- Name the label **reference-patch function localization** (not root cause).
- Log counts: patch touches no function, multi-function patches, parsing failures, cannot form a 20-pool. Keep these instances in a separate bucket for **file-level** accuracy; do not silently drop them.
- Inference input is `problem_statement` only (issue-only condition). No test patch, no `FAIL_TO_PASS`.

---

## 6. Agents [CC]

Five specialists, one pass each, temperature 0, at most 150 output tokens. Each sees a *different view* of the same instance and the shared candidate pool:

| Agent | Input view |
|---|---|
| `trace` | issue/failing-test text and traceback frames mapped to candidates (issue-reading only on SWE-bench) |
| `graph` | call-graph and import neighbors of trace frames (provided code graph if usable) |
| `spec` | docstrings, type hints, comments vs implementation for each candidate |
| `static` | AST pattern flags: comparison/boundary patterns, suspicious defaults, unreachable branches, unused variables |
| `semantic` | embedding-nearest candidates to the issue, with similarity scores |

Output format: `RANKED: C,F,A,...` plus up to 3 one-line reasons. Parse into `ranked_suspects`; reject unparseable outputs and retry once at temperature 0, then log as empty.

**Probability agent (C mode).** Single frozen Gemma. Prompt: issue, shuffled candidate block, then the notes for the subset in a **fixed canonical order** (trace, graph, spec, static, semantic; absent agents omitted), then the answer cue. Extraction per 4.1. Empty subset has no notes section.

**Forecaster (F mode).** Each specialist, given its own view and the candidates, outputs a distribution directly with the same extraction protocol (5 calls per instance).

**Sequential market (S mode).** Initial market = `p_empty`. For agent i in a fixed order, show the current top-5 market probabilities and agent i's note; extract a new distribution; reward `r_i = log p_after(G) - log p_before(G)`. Run two orders (canonical and reversed). Compare the final market to the C-mode full-evidence posterior.

All prompts live in `configs/prompts/` as files; their hashes are recorded in every row.

---

## 7. Baselines

| ID | Method | Notes |
|---|---|---|
| B0 | Random in pool | seeded, 1000 draws averaged |
| B1 | BM25 | both re-ranking the shared pool and end-to-end retrieval over whole repo |
| B2 | Embedding retrieval | same two variants |
| B3 | Ochiai spectrum-based fault localization | MarketBugs only; function-level coverage via coverage.py contexts; `ef/sqrt((ef+nf)(ef+ep))`; seeded tie-breaks |
| B4 | Single Gemma, no evidence | `p_empty` |
| B5 | Single Gemma, all 5 notes | `p_full`, canonical order |
| **B_full5** | **Compute-matched B5** | average of 5 cyclic note-order permutations of the full-evidence call |
| P_vote | Majority vote over 5 singleton top-1s | the original "equal weight" scheme |
| P_avg | Linear average of 5 singleton posteriors | |
| P_geo | Equal-weight log pool | |
| P_hedge | Hedge log pool (and linear variant) | weights learned on dev, frozen. Label: *learned supervised pool* |
| P_stack | Log-linear stacking | weights fit on dev |

Singleton posteriors for pools come from C mode subsets `{i}` (primary) and F mode (secondary). One scalar temperature, fit on dev, is applied to each pool and to B4/B5. Hedge: `w_i proportional to exp(eta * G_i)`, `G_i` = cumulative log score over the dev stream (seeded order); choose `eta` from {0.1, 0.3, 1, 3} by 2-fold CV on dev.

---

## 8. The value table (C mode)

For every instance (dev and test, each condition) compute `value_row` for all 32 subset masks. Bit order: trace=1, graph=2, spec=4, static=8, semantic=16.

- Batch all 32 calls of an instance together so the shared candidate prefix is cached.
- From the table derive: marginal credit (for a fixed order), pivotal/leave-one-out credit, and **exact Shapley** credit per agent:
  `phi_i = sum over S not containing i of |S|!(5-|S|-1)!/5! * [v(S+i) - v(S)]`.
- Check: sum of phi_i equals `v(full)`, to 1e-6.
- Expect negative credit for some agents on some instances; do not clip.
- Interpretation string used everywhere: "value of an evidence source given the fixed probability agent."

---

## 9. Experiments

### E1: primary, preregistered
- **Test:** Top-1 of **P_hedge (log pool of the five C-singleton posteriors)** versus **B_full5**, on the MarketBugs test split, `test_informed` condition. Both use the same five notes and 5 probability-agent calls; note-generation cost counted for both.
- **Analysis:** paired difference in Top-1, repo-cluster bootstrap (resample repos with replacement, then instances within repo, B = 10,000, seeded), plus per-repo table. This is the only confirmatory test.
- **Secondary (exploratory, report effect sizes and CIs, no significance claims):** Top-3/5, MRR, NLL, Brier, ECE (top-label confidence, 10 bins), by stratum, by condition (`no_trace`), gold-in-pool only vs all, SWE-bench transfer, F-mode pools, all baselines.

### E2: evidence valuation
- Report mean marginal, pivotal and Shapley credit per agent, by stratum and dataset.
- **Validation (not circular):** on dev, choose an agent order (by dev Shapley) and fixed budgets of 1, 2, 3, 5 agents. On test, compare accuracy and cost at each budget against random order, canonical order, and a cheap heuristic (trace first). Confidence-threshold stopping is secondary and only runs if dev calibration is acceptable (ECE reported on dev).

### E3: redundancy
- **Exact duplication:** add k = 1 and 2 exact copies of one distribution to P_vote, P_avg, P_geo, P_hedge, P_stack; also repeat one note in the full-evidence prompt. Run for the strongest agent and the weakest agent (by dev Shapley). Measure shift in accuracy and in weight mass on the duplicated source.
- **Same-specialist resampling:** regenerate one specialist's note at temperature 0.7 (3 samples, seeds recorded) and treat the copies as correlated but non-identical sources.
- Exact Shapley is only computed for the 5-player game.

### E4: sequential market (exploratory)
Mode S on the test split `test_informed`. Report final market top-1 / NLL vs C-mode `p_full`, order sensitivity between the two orders, and per-agent rewards `r_i`.

### E5: constrained single-function repair (only if main results finish by Wed 18:00 ET)
- MarketBugs test only. Frozen Gemma returns a full replacement for the localized function; apply by span in a **fresh checkout** per attempt; run the failing tests, then the full suite for regressions.
- Rankings compared: B0, B1, B2, B3, B_full5, P_hedge, and gold-location upper bound. Budget: 3 attempts down each ranking, identical decoding settings.
- Report fix@1, fix@3, mean attempts, and split failures into location-correct vs location-wrong. Describe it as constrained single-function repair on synthetic mutants, nothing broader.

### E6: observer LoRA (CUT)
Revive only if every P0/P1 item is done by Wed noon *and* extra repos exist for a separate train split. Training inputs and any teacher targets must come strictly from that split.

---

## 10. Statistics and reporting rules (write into `ANALYSIS_PLAN.md` before any test result exists)

- State the primary test, the datasets, the metric, the cluster bootstrap, and everything else labeled exploratory.
- Report cluster-bootstrap CIs and instance-level CIs side by side. With about 7 test repos, expect wide cluster intervals; do not switch to instance-level to get a cleaner answer.
- Report both "gold in pool" and unconditional results everywhere.
- Report the actual strata distribution; do not rebalance.
- Report every exclusion with a count and reason.
- Include a per-repo breakdown table for every headline result.
- Include total cost (note generation + all probability-agent calls) per method.
- Negative or null results get the same tables as positive ones.

---

## 11. Phases, tasks and gates

### Phase 0: Monday night (Nov 9), gate G0
- [CC] Repo scaffold, Apache-2.0, Makefile, CI with tests.
- [CC] Freeze schemas and validators (section 3).
- [CC] Write `ANALYSIS_PLAN.md` from section 10 and commit it.
- [CC] Model server on Modal; implement extraction protocol and unit tests (4.1).
- [GPT] Mutant generator v0; first repo environments; at least 20 valid mutants.
- [GPT] BM25 runner.
- [CC, parallel, **not a gate**] LoRA toolchain smoke test, only if time permits.
- [CC] Check the Kaggle Data tab for graphs or embeddings; write findings in `BLOCKERS.md`.

**G0:** 20-way probability vectors pass all extraction tests; at least 20 reproducible mutants; BM25 baseline runs end to end; CI green.

### Phase 1: Tuesday day, gate G1
- [GPT] Scale mutant generation to dev and test targets; compute strata; embeddings and call graphs; B3 Ochiai.
- [CC] SWE-bench Lite parser and sample (5.2).
- [CC] Pool builder, specialists, probability agent, value-table runner.
- [CC] Leakage test (4.4); permutation and K checks on 30 to 50 dev instances (4.3).
- [CC] **Pilot:** full 32-subset tables for at least 30 dev instances; measure tokens, latency and cost per instance; extrapolate total cost.

**G1 pass criteria**
1. `p_empty` Top-1 beats random clearly (report the numbers).
2. No leakage found.
3. Order-sensitivity within threshold, or the averaging protocol adopted.
4. Cost extrapolation for the full run fits remaining budget with at least 25% headroom.
5. Specialist notes are not near-verbatim copies of the traceback (spot-check 10).

If G1 fails: do **not** scale up. Fix the pilot, or reduce scope using the cut order (section 13).

### Phase 2: Tuesday night, gate G2 (freeze)
- [CC] Fit temperature and Hedge `eta` on dev; lock pools, prompts, scoring and the primary test.
- [CC] Write `configs/FROZEN.txt` with commit hash and config hashes.

### Phase 3: Wednesday, gate G3
- [CC] Run value tables on test (MarketBugs, then SWE-bench). Run F-mode and S-mode.
- [CC] Run baselines B0 to B5, B_full5, and all pools.
- [CC] E1, E2, E3, E4 analysis scripts and tables.
- [CC] E5 if the Wed 18:00 ET condition is met.

**G3:** at least 100 MarketBugs test instances from several distinct test repos with full tables, baselines, costs and statistics. If this fails, ship a narrower but honest result (state which parts were not run).

### Phase 4: Thursday morning, gate G4 (no new experiments)
- [CC] Re-run only failed or missing jobs.
- [CC] `make reproduce`: rebuild every headline table and figure from cached `artifacts/` with no GPU.
- [CC] Package: tables, figures, cost summary, `MarketBugs` release with generator and README, list of null results and known limitations.
- [GPT] Verify `reproduce.ipynb` runs top to bottom on a clean environment.

**G4:** headline tables reproduce from cache; all claims in the outputs match the logged numbers.

---

## 12. Budget

- Modal total about $80. Plan roughly: $40 inference (pilot + full), $15 reruns, $25 reserve. Hard-cap spend; abort jobs that exceed 2x their pilot-based estimate.
- Estimated about 50k probability-agent calls for the full plan (about 780 instance-conditions times about 60 calls). This is an estimate; replace it with the pilot measurement at G1.
- Run mutant generation, test execution (including Ochiai and E5) on local CPUs, not Modal.
- If network is unavailable in the agent's sandbox, ask the human to run clone, install and download steps and place results in `data/`.

---

## 13. Cut order (apply top to bottom if behind schedule)

1. E6 observer LoRA (already cut)
2. E5 repair
3. E4 sequential market
4. `no_trace` condition on test (keep on dev)
5. SWE-bench sample from 100 down to 60
6. Stretch repos
7. Duplicate resampling in E3 (keep exact duplication)

**Never cut:** MarketBugs test tables in C mode, baselines, E1, E2, E3 exact duplication, and at least 60 SWE-bench tasks.

---

## 14. Known risks (write mitigations into code or docs, not just here)

- **Mutants solvable from the traceback.** Mitigation: strata and `no_trace` condition.
- **Pool ceiling.** Mitigation: gold-in-pool vs unconditional, shared pool for all methods.
- **Few test repos give wide intervals.** Mitigation: cluster bootstrap; accept an inconclusive primary result if that is what the data show.
- **Probability agent miscalibration.** Mitigation: dev-fit temperature; fixed-budget stopping is primary.
- **Gemma may have seen SWE-bench.** Mitigation: report SWE-bench separately and describe it as descriptive.
- **Env/setup time dominates.** Mitigation: prefer fast pure-Python repos; cut stretch repos first.
- **Gemma 4 tooling gaps.** Mitigation: fallback to the smallest supported Gemma; record in `BLOCKERS.md`.

---

## 15. Open decisions for the human (do not guess; ask)

1. Final repo list for MarketBugs.
2. Exact Gemma 4 variant and revision to use as the frozen probability agent.
3. Whether the Kaggle-provided graphs or embeddings exist and are usable.
4. Whether the primary test stays as defined in E1.
5. Any change to the 15% order-sensitivity threshold, the 3-statement minimum, or the caps in 5.1.

---

## 16. Definition of done (experiment phase)

- [ ] Schemas validated; CI green; leakage test passes on the final prompts.
- [ ] `FROZEN.txt` committed before any test-split result.
- [ ] MarketBugs dev/test and SWE-bench sample built, with all exclusions counted.
- [ ] Value tables complete for test (MarketBugs, SWE-bench subset); exact Shapley sums verified.
- [ ] E1 primary test computed exactly as preregistered; E2 and E3 complete; E4 and E5 complete or explicitly marked not run.
- [ ] Per-repo, per-stratum, gold-in-pool and cost tables exist for every headline result.
- [ ] `make reproduce` rebuilds all tables from `artifacts/` without GPU.
- [ ] `RESULTS_LEDGER.md` lists each reported number with its script, commit and artifact hash.
- [ ] `BLOCKERS.md` lists every unverified item and every deviation from this plan.
