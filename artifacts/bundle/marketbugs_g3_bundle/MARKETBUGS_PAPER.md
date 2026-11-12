# Evidence Markets for Small-Model Fault Localization: A Preregistered Held-Out Test

**Repository:** https://github.com/raunak2007/marketbugs (all numbers below link to artifacts in `artifacts/`).

## Abstract

Can a small language model localize bugs better when several specialist "participants" contribute evidence and
their beliefs are aggregated, prediction-market style, into one probability forecast? We build MarketBugs, 330
validated single-function mutants from 10 permissively licensed Python libraries, split by repository into
development (3 repos, 120 bugs) and held-out test (7 repos, 210 bugs). Five specialists (trace, call graph,
specification, static patterns, semantic similarity) write short notes about 20 retrieved candidate functions; a
frozen Gemma 4 E4B probability agent turns any subset of notes into a distribution over the candidates. Our
preregistered test compares a development-trained Hedge pool of the five single-note forecasts (P_hedge) with a
matched five-call full-evidence baseline (B_full5) on the untouched test split. On 210 held-out bugs, P_hedge reached 63.8% Top-1 (134/210) and B_full5 64.8% (136/210): a paired difference of −0.95 percentage points (95% repository-cluster bootstrap CI −4.3 to +2.9), inconclusive. The two methods chose the same function on 202 of 210 bugs. Adding all five notes to a single call moved Top-1 from 62.4% to 64.8% (+2.4 points, CI −3.3 to +8.1), also inconclusive, and the calibration advantage pooling showed on development data reversed on test. Gold functions were
retrieved into the candidate pool for 81.9% (172/210) of test bugs, which caps every method. For this small model, the decisive factors were retrieval and candidate ordering, not how specialist evidence was aggregated: the same model, given the same issue and candidates, mostly reached the same answer regardless of the notes or the pooling rule. Seven
repository clusters limit precision, and mutants are not natural bugs.

## 1. Introduction

Repository-level fault localization asks which function to change. Small open models are cheap enough to call many
times, which invites an old idea: instead of one model reading everything, let specialists each contribute a partial
view, and aggregate their beliefs. Prediction markets are the canonical example of such aggregation: participants with
different information move a shared probability until it reflects the pooled evidence.

We use "evidence market" in the operational sense of aggregating specialized information into a shared probability
forecast, rather than claiming an incentive-compatible financial prediction market. Our participants are prompted
specialists; the "price" is a probability distribution over candidate functions; aggregation rules are fixed or
learned on development data. There is no trading, currency, market-clearing price or strategic incentive.

The question is narrow and testable: **does information-aware pooling of specialist evidence beat a single model
call that sees the same evidence?** We answer it with one preregistered, held-out comparison, frozen before any test
inference, and report it whatever its sign.

Contributions: (1) MarketBugs, a repository-split mutant benchmark with evaluator-only gold and validated failures;
(2) a coalition protocol that turns any subset of specialist notes into a calibrated-by-construction distribution
over a fixed candidate pool; (3) a preregistered held-out comparison with a repository-cluster bootstrap, with all
code, frozen parameters and per-instance results public.

## 2. Related work

*Spectrum-based fault localization* ranks code by coverage of failing vs passing tests (Tarantula, Jones & Harrold
2005; Ochiai, Abreu et al. 2007; survey: Wong et al. 2016). *Mutation-based benchmarks* such as Defects4J (Just et al.
2014) motivate seeded single-function faults with reproducible failing tests. *LLM localization and repair* on
SWE-bench (Jimenez et al. 2024) often uses staged retrieval and ranking (e.g. Agentless, Xia et al. 2024; AutoFL,
Kang et al. 2024). *Forecast aggregation* combines probabilistic opinions by linear or logarithmic pooling (Genest &
Zidek 1986); online weighting by exponentiated cumulative log score is the Hedge algorithm (Freund & Schapire 1997).
*Prediction markets* aggregate dispersed information into prices (Wolfers & Zitzewitz 2004); market scoring rules
(Hanson 2003) reward participants by the change in log score they cause, which our sequential mode borrows without
any incentive claim. *Shapley values* (Shapley 1953) attribute a coalition's value to its members.

## 3. MarketBugs

We screened 16 Python libraries for permissive licenses, pure-Python code, deterministic suites and fast tests; 10
passed. A seeded split assigned 3 repositories to development (cachetools, intervaltree, sortedcontainers) and 7 to
test (attrs, boltons, dateutil, isodate, more-itertools, tomli, toolz), fixed before any result.

Each instance is a minimal AST-span mutant of one function (comparison flip, operator swap, off-by-one, argument
swap, condition removal, wrong variable, statement deletion, return change). A mutant is kept only if the original
suite passes, the mutant fails at least one test identically over three runs, exactly one function changed and that
function has at least three statements. Instances are capped at 40 per development repo and 30 per test repo:
120 + 210 = 330. Issue text is templated from the failing test (name, assertion, traceback). Gold location, patch and
operator live in evaluator-only fields that inference code cannot read (whitelisted inference views plus leakage
tests over every prompt builder). Strata: 200 *direct* (gold in traceback), 41 *near* (one call hop), 89 *far*.

## 4. An evidence market over candidate functions

**Candidate pool.** For each bug, 20 functions are retrieved by interleaving BM25, embedding similarity and
call-graph neighbours of traceback frames, then shuffled with a seed that does not encode rank. The gold function is
never inserted: if retrieval misses it, every method scores zero on that bug.

**Participants.** Five specialists see different views and write a ranked note (temperature 0, ≤150 tokens):
*trace* (failing test and frames), *graph* (call/import neighbours), *spec* (docstrings and types vs code), *static*
(AST risk patterns), *semantic* (embedding similarity to the issue).

**The shared forecast.** A frozen probability agent (Gemma 4 E4B-it, pinned revision, bf16) reads the issue, the
candidate block and any subset of notes, and ends at an answer cue; we softmax the logits of the 20 single-token
labels. This gives a distribution over candidates for each of the 32 note subsets (*Mode C, evidence-coalition
valuation*), and a value for each coalition: v(S) = log p_S(gold) − log p_∅(gold).

**Aggregation rules.**
- **B_full5** (matched baseline): one call with all five notes, averaged over the five cyclic note orders (5 calls).
- **P_hedge** (*learned supervised probability pool*, not a market mechanism): a weighted logarithmic pool of the
  five single-note forecasts (5 calls), with weights w_i ∝ exp(η G_i) from cumulative development log scores.
- **Mode S** (*sequential evidence market*): participants update a shared forecast one at a time, seeing the current
  top-5 probabilities; each is "rewarded" by its change in log score. Only this mode resembles successive market
  updates; it is exploratory and was not run on test.

Both primary methods use the same five notes and five probability-agent calls per candidate order, so the comparison
isolates *how evidence is combined*, not how much is bought.

**Order control.** On development data 25.7% (G1) to 33% (G2) of Top-1 picks changed with the candidate display
order, above our preregistered 15% threshold, so every method averages log-probabilities over two fixed candidate
permutations (*avg2*), aligned by canonical candidate ID.

## 5. Preregistered protocol

Frozen before any test inference (`configs/FROZEN.txt`, `artifacts/frozen_params.json`, code commit
`bda0716`): model revision, prompts, K = 20, avg2, η = 0.1 (2-fold CV over {0.1, 0.3, 1, 3}), Hedge weights
trace .380, semantic .236, spec .149, static .123, graph .113, and per-method temperatures, all fit on the 120
development bugs only. The test runner refuses to start if code, configs, prompts or split changed since the freeze.

**Primary statistic:** mean paired Top-1 difference P_hedge − B_full5 over all 210 test bugs (retrieval misses count
as wrong for both). **Uncertainty:** 10,000-replicate two-stage repository-cluster bootstrap (resample the 7 repos,
then bugs within each sampled repo copy), seed 20261110, percentile 95% CI. A CI spanning zero is inconclusive, not
evidence of equality. Everything else is exploratory.

**Pre-test amendment (disclosed).** Two development bugs had runaway doctest output (29k and 4.8M tokens). Before the
freeze we capped issue text at 4,096 tokens (first 1,024 + last 3,072); this changed exactly those two development
prompts and applies identically to test.

**Scope cut (disclosed).** To fit compute, test inference computed only the calls the primary comparison and its
context need (no-note, five single-note and full-note forecasts, five note orders, two candidate orders): 22
probability-agent calls and 5 note generations per bug. Mode S, placebo notes, the 32-subset table and SWE-bench
transfer were not run on test.

## 6. Results

### 6.1 Primary held-out comparison

P_hedge did not beat the matched baseline. Over all 210 held-out bugs (retrieval misses counted as wrong), P_hedge was correct on 134 (63.8%) and B_full5 on 136 (64.8%). The mean paired difference was **−0.95 percentage points**, with a **95% two-stage repository-cluster bootstrap CI of [−4.3, +2.9]** (10,000 replicates, seed 20261110). The interval spans zero, so the result is **inconclusive**: it neither shows an advantage nor establishes equivalence, and it rules out a P_hedge gain larger than about 3 points on this population of repositories. The striking feature is agreement: the methods disagreed on only 8 bugs (P_hedge right and B_full5 wrong on 3; the reverse on 5). An instance-level bootstrap (diagnostic only, ignoring repository clustering) gives [−3.8, +1.4].

| | P_hedge | B_full5 |
|---|---|---|
| Top-1, all 210 | **134 (63.8%)** | **136 (64.8%)** |
| Top-1, gold in pool (172) | 77.9% | 79.1% |
| Top-3 / Top-5 | 74.8% / 78.1% | 75.7% / 76.7% |
| MRR | .702 | .707 |
| log-loss, tempered (gold in pool) | .814 | .749 |
| paired W / L / T vs the other | 3 / 5 / 202 | 5 / 3 / 202 |
| probability-agent calls per bug | 10 | 10 |

### 6.2 Retrieval ceiling and conditional accuracy

The 20-candidate pool contained the gold function for **172 of 210** bugs (81.9%); the remaining 38 (18.1%) are failures for every method, so no aggregation rule can exceed 81.9%. Conditional on gold being in the pool, P_hedge ranked it first 77.9% of the time and B_full5 79.1%. Retrieval misses therefore cost more accuracy (18 points) than the entire spread between methods (≤ 3.8 points). Accuracy also tracked how visible the bug was in the trace: on *direct* bugs (gold in the traceback, n = 118) both methods were about 83% correct, against about 40% on *near* (n = 28) and *far* (n = 64) bugs.

### 6.3 Per repository

| repository | n | P_hedge Top-1 | B_full5 Top-1 | difference |
|---|---|---|---|---|
| attrs | 30 | .367 | .367 | 0 |
| boltons | 30 | .633 | .667 | −.033 |
| dateutil | 30 | .633 | .667 | −.033 |
| isodate | 30 | .633 | .633 | 0 |
| more-itertools | 30 | .700 | .733 | −.033 |
| tomli | 30 | .833 | .767 | +.067 |
| toolz | 30 | .667 | .700 | −.033 |

P_hedge was behind in four repositories, tied in two and ahead in one; every per-repository difference is one or two bugs out of 30. Repository difficulty varied far more (attrs .37 to tomli .77–.83) than the method gap.

### 6.4 Context methods (exploratory)

Without notes, the probability agent already reached 62.4% (131/210). All five notes in one call gave 64.8% (+2.4 points; cluster CI [−3.3, +8.1], inconclusive); averaging over five note orders (B_full5) did not change Top-1 relative to the single canonical order (both 136/210). Single-note forecasts ranged from 61.4% (trace) to 66.2% (static); this ranking is post hoc and should not be read as a finding. Development data had suggested pooling mainly improves calibration; on test the reverse held (tempered log-loss .749 for B_full5 vs .814 for P_hedge). Candidate display order still changed the Top-1 pick on 11–23% of bugs depending on the note subset, so the order averaging remained necessary.

### 6.5 Development results (exploratory, not validation)

On the 120 development bugs (the same data used to fit η, weights and temperatures), Top-1 was .617 with no notes,
.675 with all notes and .692 for B_full5; P_hedge reached .683 with the best development log-loss (.660 vs .723 for
the single full-evidence call). P_hedge was nearly identical to an equal-weight pool (.663), and η sat at the lowest
grid value. These numbers motivated, but cannot confirm, the held-out test.

### 6.6 Cost

Both primary methods use five note generations and ten probability-agent calls per bug (five per candidate order). Estimated GPU cost was about $0.020 per bug for either method ($0.015 of it for the shared notes). The full held-out run (22 probability-agent calls and 5 notes per bug) metered about $7.5 on Modal A10G GPUs; all reported costs are provider-metered totals or conservative wall-clock estimates as labelled in the artifacts.

## 7. Discussion

The preregistered test returned a null with a usefully narrow ceiling: whatever advantage information-aware pooling has here is unlikely to exceed about 3 points of Top-1, and the point estimate slightly favours the simpler single call. Three observations explain why aggregation had so little room. First, **the shared forecaster dominates**. Every participant's note is read by the same small model; 202 of 210 decisions were identical across rules, and notes added only 2.4 points over no notes. Specialists built from one model appear to contribute correlated, not independent, information, which is exactly the condition under which pooling, market-style or otherwise, cannot help. Second, **retrieval and ordering dominate aggregation**. Retrieval misses cost 18 points and candidate order flipped up to 23% of picks, both far larger than the method gap. Third, **development gains did not transfer**. The development set suggested better calibration from pooling and a learned weighting favouring the trace specialist; on held-out repositories the calibration advantage reversed and the learned weights gave no accuracy gain, a reminder that three development repositories are a thin basis for fitting even five weights. For practitioners, the evidence-market framing is still useful as a design vocabulary (independent signals, a shared forecast, explicit aggregation rules), but its benefit depends on participants having genuinely different information. Diverse models, executed tests or coverage data are more promising sources of that diversity than prompts to one model.

## 8. Limitations

Seven test repositories make the cluster CI wide; mutants are not natural bugs, and single-function mutants with
templated issues favour trace evidence. One small model (Gemma 4 E4B) serves every role, so "specialists" share its
blind spots. Candidate-order and note-order sensitivity are large, which is why we average over orders; the averaging
itself is a design choice. Our "market" has no incentives or trading: it is an aggregation scheme, and Hedge is a
supervised pool. Development and held-out calls ran on Modal A10G GPUs in bf16; bf16 batch composition can change
low-order bits (99.5% Top-1 agreement on identical prompts across runs).

## 9. Conclusion

We built a market-inspired fault localizer in which specialist participants contribute evidence to a shared probability forecast, and tested its central claim with a frozen, preregistered held-out comparison. On 210 bugs from seven unseen repositories, the learned Hedge pool did not outperform a matched full-evidence baseline (−0.95 points, 95% repository-cluster CI [−4.3, +2.9]), and the two agreed on 96% of bugs. The limiting factors were retrieval (18% misses) and candidate ordering, not aggregation. A credible evidence market for code needs participants with independent information; with one small model in every role, the market mostly echoes its own forecaster. All data, frozen parameters, per-instance predictions and code are public for re-analysis.

## References

- Abreu, R., Zoeteweij, P., van Gemund, A. (2007). On the accuracy of spectrum-based fault localization. TAICPART-MUTATION.
- Freund, Y., Schapire, R. (1997). A decision-theoretic generalization of on-line learning and an application to boosting. JCSS 55(1).
- Genest, C., Zidek, J. (1986). Combining probability distributions: a critique and an annotated bibliography. Statistical Science 1(1).
- Hanson, R. (2003). Combinatorial information market design. Information Systems Frontiers 5(1).
- Jimenez, C. et al. (2024). SWE-bench: Can language models resolve real-world GitHub issues? ICLR.
- Jones, J., Harrold, M. J. (2005). Empirical evaluation of the Tarantula automatic fault-localization technique. ASE.
- Just, R., Jalali, D., Ernst, M. (2014). Defects4J: a database of existing faults to enable controlled testing studies for Java programs. ISSTA.
- Kang, S., An, G., Yoo, S. (2024). A quantitative and qualitative evaluation of LLM-based explainable fault localization. FSE.
- Shapley, L. (1953). A value for n-person games. Contributions to the Theory of Games II.
- Wolfers, J., Zitzewitz, E. (2004). Prediction markets. Journal of Economic Perspectives 18(2).
- Wong, W. E., Gao, R., Li, Y., Abreu, R., Wotawa, F. (2016). A survey on software fault localization. IEEE TSE 42(8).
- Xia, C. S., Deng, Y., Dunn, S., Zhang, L. (2024). Agentless: demystifying LLM-based software engineering agents. arXiv:2407.01489.
