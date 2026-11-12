# MarketBugs: What Makes Evidence Aggregation Reliable for Small-Model Software Debugging?

[**Paper**](paper/MARKETBUGS_PAPER.md) · [**Held-out results**](artifacts/validation/heldout_v1/RESULTS.md) · [**Preregistration**](ANALYSIS_PLAN.md) · [**Decision log**](BLOCKERS.md) · [**Project log**](docs/PROGRESS.md)

Can a small open model find the buggy function in a repository more reliably if several specialists each contribute
a piece of evidence and their beliefs are pooled into one forecast? We built a market-inspired fault localizer
around Gemma 4 E4B, wrote down the main test before looking at any held-out data, and ran it on 210 bugs from seven
repositories the system had never seen.

**The short answer is no.** The learned pool and the simpler baseline had the same correctness on 202 of 210 bugs
(they picked the same function on 193), and the difference between them is indistinguishable from zero.

## Main result

Preregistered comparison on the held-out split (210 bugs, 7 repositories, `test_informed` condition):

| | P_hedge (learned pool) | B_full5 (matched baseline) |
|---|---|---|
| Top-1, all 210 bugs | 134 (63.8%) | 136 (64.8%) |
| Top-1, gold function in the candidate pool (172 bugs) | 77.9% | 79.1% |
| Paired wins / losses / ties | 3 / 5 / 202 | |

Paired Top-1 difference, P_hedge minus B_full5: **−0.95 percentage points**, 95% two-stage repository-cluster
bootstrap CI **[−4.3, +2.9]** (10,000 replicates, seed 20261110). The interval includes zero, so the test is
inconclusive. It does not show the methods are equivalent, but it makes a pooling gain much above 3 points unlikely
on this population of repositories.

A few other things stood out:

- Retrieval was the real bottleneck. The gold function made it into the 20 candidates for 172 of 210 bugs
  (81.9%), so 38 bugs were lost before any aggregation happened.
- The specialist notes helped less than we expected. One Gemma call with all five notes beat a call with none by
  2.4 points (CI −3.3 to +8.1), which is also inconclusive.
- On development data, pooling looked better calibrated. On held-out data it was the other way around (tempered
  log-loss 0.814 for P_hedge against 0.749 for B_full5).
- Shuffling the order of the candidates still changed the top pick on 11% to 23% of bugs, depending on which notes
  the model saw.

Our reading is that five "specialists" built from the same small model are not independent enough for any pooling
rule to have much to work with. The paper discusses this in Section 7.

## How it works

Each bug comes with a failing test and a traceback. The pipeline:

1. **Retrieves 20 candidate functions** with BM25, embedding similarity and call-graph neighbours of traceback
   frames. The gold function is never inserted, so a retrieval miss counts as a failure for every method.
2. **Asks five specialists for a short note** about the candidates. Each sees a different view of the bug: the
   failing test and frames (trace), call and import neighbours (graph), docstrings and types (spec), AST risk
   patterns (static), and embedding similarity (semantic).
3. **Turns notes into a forecast.** A frozen Gemma "probability agent" reads the issue, the candidates and any
   subset of notes, and we read off a probability for each candidate from the logits of 20 single-token labels.
4. **Combines the evidence** in one of several ways:
   - **B_full5**: one call with all five notes, averaged over the five cyclic note orders.
   - **P_hedge**: a weighted log pool of the five single-note forecasts, with weights learned on development data.
     This is a learned supervised probability pool, not a market mechanism.
   - **Mode S**, the sequential evidence market: specialists update a shared forecast one at a time and are scored
     by how much they move it toward the right answer. This mode is exploratory and was not run on the held-out
     split.

We use "evidence market" in the operational sense of aggregating specialized information into a shared probability
forecast. Nothing here trades, clears a price, or gives participants incentives.

Every method averages log-probabilities over two fixed candidate orders, because on development data more than 15%
of top picks changed with the display order (our preregistered threshold).

## The MarketBugs benchmark

MarketBugs has 330 single-function mutants from 10 permissively licensed Python libraries. Each mutant is a minimal
AST edit that makes at least one test fail the same way over three runs. Repositories are split before any results,
so no test repository is ever seen during development:

| split | repositories | bugs |
|---|---|---|
| development | cachetools, intervaltree, sortedcontainers | 120 (40 each) |
| held-out test | attrs, boltons, dateutil, isodate, more-itertools, tomli, toolz | 210 (30 each) |

Gold locations, patches and mutation operators live in evaluator-only fields. Inference code reads instances through
a whitelisted view, and a leakage test scans every prompt builder. The data is in `data/marketbugs/{dev,test}/`.

## Reproducing the results

Everything in the paper can be rebuilt from the committed artifacts on a CPU:

```bash
make test        # unit tests: schemas, extraction, no-leak, freeze gate, primary-only equivalence
make validate    # schema-validate every dataset row

# Score the held-out run (gold is attached only here, after an audit that never reads it)
.venv/bin/python -m src.analysis.evaluate_test --tag heldout_v1 --frozen artifacts/frozen_params.json \
    --require-n 210 --output-dir artifacts/validation/heldout_v1
.venv/bin/python -m src.analysis.bootstrap_test --input artifacts/validation/heldout_v1/per_instance.csv \
    --seed 20261110 --replicates 10000 --output artifacts/validation/heldout_v1/confidence_intervals.json
```

Re-running inference needs a GPU. We used Modal A10G instances in bf16:

```bash
.venv/bin/modal deploy -m src.llm.modal_server
.venv/bin/python -m src.agents.run_test --input-dir data/marketbugs/test --tag heldout_v1 --cond test_informed \
    --protocol avg2 --arms real --seed 20261111 --backend modal --workspace <your-workspace> \
    --workspace-cap-usd <cap> --max-total-usd 45 --no-mode-s --primary-only --workers 6 --resume
```

The runner refuses to start unless `configs/FROZEN.txt` and `artifacts/frozen_params.json` are committed and no
code, prompt, config or split file has changed since the freeze. Paid runs must name a Modal workspace and a monthly
dollar cap. The held-out run cost about $7.50 on Modal.

The cloned benchmark repositories (`data/**/repos/`) are git-ignored. `src.gen.eval_repos` rebuilds them.

## Repository layout

```
paper/        the paper (MARKETBUGS_PAPER.md)
configs/      pinned model revisions, repo split, prompts, FROZEN.txt (the frozen evaluation contract)
schemas/      JSON Schemas for instances, candidates, notes and value rows
src/gen/      MarketBugs generator: repo screening, mutants, strata, trace sanitizing
src/cands/    function index, BM25 / embedding / call-graph retrieval, 20-candidate pools
src/agents/   specialists, probability agent, dev pilot driver, held-out runner
src/llm/      Gemma forward-pass backend, Modal server, spend guard, tokenizer utilities
src/market/   pooling, Hedge, Shapley, sequential market
src/analysis/ dev freeze, held-out evaluator, repository-cluster bootstrap
data/         marketbugs/{dev,test}/<repo>.jsonl and evaluator-only eval_only.jsonl
artifacts/    run data, frozen parameters, validation outputs, cost ledger, results bundle
docs/         project log
```

## Limitations

Seven test repositories give only seven clusters for the bootstrap, so the interval is wide. Mutants are not
natural bugs, and templated issues with tracebacks favour trace evidence. One small model plays every role. Mode S,
the placebo-note control, the full 32-subset coalition table and SWE-bench transfer were not run on the held-out
split. The paper lists every deviation from the plan, and `BLOCKERS.md` has the full record.

## Citation

```bibtex
@misc{marketbugs2026,
  title  = {MarketBugs: What Makes Evidence Aggregation Reliable for Small-Model Software Debugging?},
  author = {Mondal, Raunak},
  year   = {2026},
  url    = {https://github.com/raunak2007/marketbugs}
}
```

## License

Apache 2.0, see [LICENSE](LICENSE). Each benchmark instance records the license of the library it came from.
