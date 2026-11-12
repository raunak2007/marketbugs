# Evidence Markets for Small-Model Fault Localization: A Preregistered Held-Out Test

Code, data and per-bug results: https://github.com/raunak2007/marketbugs (every number here comes from `artifacts/`).

## Abstract

Can a small language model localize bugs better when several specialists contribute evidence and their beliefs are
pooled, prediction-market style, into one probability forecast? We built MarketBugs, 330 validated single-function
mutants from 10 permissively licensed Python libraries, split by repository into a development set (3 repositories,
120 bugs) and a held-out test set (7 repositories, 210 bugs). Five specialists (trace, call graph, specification,
static patterns, semantic similarity) write short notes about 20 retrieved candidate functions, and a frozen Gemma 4
E4B probability agent turns any subset of those notes into a distribution over the candidates. Before touching the
test set, we registered one comparison: a Hedge pool of the five single-note forecasts trained on development data
(P_hedge) against a matched five-call baseline that reads all the notes at once (B_full5). On the 210 held-out bugs,
P_hedge got 134 right (63.8% Top-1) and B_full5 got 136 (64.8%). The paired difference is −0.95 percentage points,
with a 95% repository-cluster bootstrap interval of −4.3 to +2.9, so the test is inconclusive. The two methods chose
the same function on 202 of the 210 bugs. Adding all five notes to a single call moved Top-1 from 62.4% to 64.8%
(+2.4 points, interval −3.3 to +8.1), and the calibration advantage pooling had shown on development data reversed
on the test set. The gold function reached the candidate pool for only 172 of 210 bugs (81.9%), which caps every
method. For this model, retrieval and candidate ordering mattered more than how the specialist evidence was
combined. Seven repository clusters limit precision, and mutants are not natural bugs.

## 1. Introduction

Repository-level fault localization asks which function to change. Small open models are cheap enough to call many
times, which makes an old idea attractive: rather than having one model read everything, let several specialists
each look at part of the evidence and then aggregate what they believe. Prediction markets are the familiar example
of this. Traders with different information push a shared price until it reflects what they collectively know.

We borrow the vocabulary, not the economics. We use "evidence market" in the operational sense of aggregating
specialized information into a shared probability forecast, rather than claiming an incentive-compatible financial
prediction market. Our participants are prompted specialists, the "price" is a probability distribution over
candidate functions, and the aggregation rules are fixed in advance or learned on development data. There is no
trading, no currency, no market-clearing price and no strategic incentive anywhere in the system.

That leaves a narrow question we can test: does information-aware pooling of specialist evidence beat a single
model call that sees the same evidence? We answer it with one preregistered comparison on held-out repositories,
frozen before any test inference. We contribute MarketBugs, a repository-split mutant benchmark with evaluator-only
gold labels; a coalition protocol that turns any subset of notes into a distribution over a fixed candidate pool;
and the held-out comparison, with all code and per-bug results public.

## 2. Related work

Spectrum-based fault localization ranks code by how often failing and passing tests execute it (Tarantula, Jones &
Harrold 2005; Ochiai, Abreu et al. 2007; see the survey by Wong et al. 2016). Mutation-based benchmarks such as
Defects4J (Just et al. 2014) motivate seeded single-function faults with reproducible failing tests. Work on LLM
localization and repair, much of it evaluated on SWE-bench (Jimenez et al. 2024), often uses staged retrieval and
ranking (for example Agentless, Xia et al. 2024, and AutoFL, Kang et al. 2024). Forecast aggregation combines
probabilistic opinions with linear or logarithmic pools (Genest & Zidek 1986), and weighting experts by their
exponentiated cumulative log score is the Hedge algorithm (Freund & Schapire 1997). Prediction markets aggregate
dispersed information into prices (Wolfers & Zitzewitz 2004), and market scoring rules (Hanson 2003) reward the
change in log score a participant causes, which our sequential mode borrows. Shapley values (Shapley 1953) divide a coalition's value among its members.

## 3. MarketBugs

We screened 16 Python libraries for permissive licenses, pure-Python code, deterministic test suites and fast tests,
and 10 passed. A seeded split, fixed before any results existed, put 3 repositories in development (cachetools,
intervaltree, sortedcontainers) and 7 in test (attrs, boltons, dateutil, isodate, more-itertools, tomli, toolz).

Each instance is a minimal AST-span mutation of one function: a flipped comparison, a swapped operator, an
off-by-one constant, swapped arguments, a removed condition, a wrong variable, a deleted statement or a changed
return value. We kept a mutant only if the original suite passes, the mutant fails at least one test identically
over three runs, exactly one function changed, and that function has at least three statements. With caps of 40 per
development repository and 30 per test repository, that gives 120 + 210 = 330 bugs. The issue text is generated
from the failing test (its name, the assertion and the traceback). The gold location, patch and operator sit in
evaluator-only fields. Inference code can only read instances through a whitelisted view, and leakage tests check
every prompt builder. By how visible the bug is in the trace, 200 bugs are direct (the gold function appears in the
traceback), 41 are near (one call away) and 89 are far.

## 4. An evidence market over candidate functions

**Candidate pool.** For each bug we retrieve 20 functions by interleaving BM25, embedding similarity and call-graph
neighbours of the traceback frames, then shuffle them with a seed that does not encode retrieval rank. We never
insert the gold function. If retrieval misses it, every method scores zero on that bug.

**Participants.** Five specialists each see a different view and write a ranked note (temperature 0, at most 150
tokens). The trace specialist sees the failing test and frames; graph sees call and import neighbours; spec compares
docstrings and type hints with the code; static looks for AST risk patterns; semantic sees embedding similarity to
the issue.

**The shared forecast.** A frozen probability agent (Gemma 4 E4B-it at a pinned revision, in bf16) reads the issue,
the candidate block and any subset of the notes, and stops at an answer cue. We take a softmax over the logits of the
20 single-token candidate labels. This gives a distribution over candidates for each of the 32 note subsets, which we
call Mode C, evidence-coalition valuation. Each coalition S gets a value v(S) = log p_S(gold) − log p_∅(gold).

**Aggregation rules.**
- B_full5, the matched baseline, makes one call with all five notes, averaged over the five cyclic note orders (5
  calls).
- P_hedge is a weighted logarithmic pool of the five single-note forecasts (5 calls), with weights w_i ∝ exp(η G_i)
  computed from cumulative development log scores. It is a learned supervised probability pool, not a market
  mechanism.
- Mode S, the sequential evidence market, has participants update a shared forecast one at a time while seeing its
  current top five probabilities, and scores each one by the change in log score it causes. It is the only mode that
  resembles successive market updates. It is exploratory and was not run on the test set.

The two primary methods use the same five notes and the same number of probability-agent calls for each candidate
order. The comparison therefore isolates how the evidence is combined, not how much of it is bought.

**Order control.** On development data, 25.7% (in the 30-bug pilot) and then 33% (on all 120 bugs) of Top-1 picks
changed when we changed the order in which candidates were displayed. Both are above our preregistered 15%
threshold, so every method averages log-probabilities over two fixed candidate orders, aligned by candidate ID.

## 5. Preregistered protocol

We froze the following before any test inference, in `configs/FROZEN.txt` and `artifacts/frozen_params.json` at code
commit `bda0716`: the model revision, prompts, K = 20, two-order averaging, η = 0.1 (chosen by 2-fold
cross-validation from {0.1, 0.3, 1, 3}), the Hedge weights (trace .380, semantic .236, spec .149, static .123, graph
.113) and one temperature per method. All of them were fit on the 120 development bugs only. The test runner refuses
to start if code, configs, prompts or the split have changed since the freeze.

The primary statistic is the mean paired Top-1 difference, P_hedge minus B_full5, over all 210 test bugs, with
retrieval misses counted as wrong for both. For uncertainty we use a 10,000-replicate two-stage repository-cluster
bootstrap: resample the 7 repositories, then resample bugs within each sampled copy, with seed 20261110, and take the
2.5th and 97.5th percentiles. An interval that spans zero is inconclusive; it is not evidence of equality. Everything
else in this paper is exploratory.

**A change made before the freeze.** Two development bugs had runaway doctest output in their issue text (29k and
4.8M tokens). We capped issue text at 4,096 tokens, keeping the first 1,024 and the last 3,072. This changed exactly
those two development prompts, and the same rule applies to the test set.

**A cut in scope.** To fit our compute budget, test inference ran only the calls needed for the primary comparison
and its context: the no-note, single-note and all-note forecasts, the five note orders and the two candidate orders.
That is 22 probability-agent calls and 5 note generations per bug. We did not run Mode S, the placebo-note control,
the full 32-subset table or the SWE-bench transfer on the test set.

## 6. Results

### 6.1 Primary held-out comparison

P_hedge did not beat the matched baseline. Over all 210 held-out bugs, with retrieval misses counted as wrong,
P_hedge was correct on 134 (63.8%) and B_full5 on 136 (64.8%). The mean paired difference is −0.95 percentage points,
with a 95% two-stage repository-cluster bootstrap interval of [−4.3, +2.9] (10,000 replicates, seed 20261110). The
interval includes zero, so the result is inconclusive. It neither shows an advantage nor establishes equivalence,
but it does make a P_hedge gain of more than about 3 points unlikely on this population of repositories.

The two methods disagreed on only 8 bugs: P_hedge was right and B_full5 wrong on 3, and the reverse on 5. An instance-level bootstrap (a diagnostic that ignores clustering) gives [−3.8, +1.4].

| | P_hedge | B_full5 |
|---|---|---|
| Top-1, all 210 | 134 (63.8%) | 136 (64.8%) |
| Top-1, gold in pool (172) | 77.9% | 79.1% |
| Top-3 / Top-5 | 74.8% / 78.1% | 75.7% / 76.7% |
| MRR | .702 | .707 |
| Log-loss, tempered (gold in pool) | .814 | .749 |
| Paired W / L / T against the other | 3 / 5 / 202 | 5 / 3 / 202 |
| Probability-agent calls per bug | 10 | 10 |

### 6.2 Retrieval ceiling and conditional accuracy

The 20-candidate pool contained the gold function for 172 of the 210 bugs (81.9%). The other 38 (18.1%) are failures
for every method, so no aggregation rule can score above 81.9%. When gold was in the pool, P_hedge ranked it first
77.9% of the time and B_full5 79.1%. Retrieval misses cost more accuracy (18 points) than the whole spread between
methods (at most 3.8 points). Accuracy also tracked how visible the bug was in the trace. On direct bugs (n = 118)
both methods were about 83% correct; on near (n = 28) and far (n = 64) bugs they were about 40%.

### 6.3 Per repository

| Repository | n | P_hedge Top-1 | B_full5 Top-1 | Difference |
|---|---|---|---|---|
| attrs | 30 | .367 | .367 | 0 |
| boltons | 30 | .633 | .667 | −.033 |
| dateutil | 30 | .633 | .667 | −.033 |
| isodate | 30 | .633 | .633 | 0 |
| more-itertools | 30 | .700 | .733 | −.033 |
| tomli | 30 | .833 | .767 | +.067 |
| toolz | 30 | .667 | .700 | −.033 |

Every per-repository difference is one or two bugs out of 30, while difficulty varied far more between
repositories (from .37 on attrs to .77 and .83 on tomli) than between methods.

### 6.4 Context methods (exploratory)

With no notes at all, the probability agent already got 62.4% (131/210). All five notes in one call gave 64.8%, a
gain of 2.4 points (cluster interval [−3.3, +8.1], inconclusive). Averaging over five note orders did not change
Top-1 compared with the single canonical order; both got 136 of 210. Single-note forecasts ranged from 61.4% (trace)
to 66.2% (static); that ranking was not planned and is not a finding. The development
data had suggested that pooling mainly helps calibration. On the test set it went the other way, with a tempered
log-loss of .749 for B_full5 and .814 for P_hedge. The order in which candidates were displayed still changed the
Top-1 pick on 11% to 23% of bugs, depending on the note subset, so the order averaging was still needed.

### 6.5 Development results (exploratory, not validation)

On the 120 development bugs, the same data used to fit η, the weights and the temperatures, Top-1 was .617 with no
notes, .675 with all notes and .692 for B_full5. P_hedge reached .683 and had the best development log-loss (.660,
against .723 for the single full-evidence call). It was nearly identical to an equal-weight pool (.663), and η landed
on the lowest value in the grid. They motivated the held-out test but confirm nothing.

### 6.6 Cost

Both primary methods use five note generations and ten probability-agent calls per bug (five per candidate order).
We estimate GPU cost at about $0.020 per bug for either method, of which $0.015 goes to the shared notes. The full
held-out run (22 probability-agent calls and 5 notes per bug) metered about $7.5 on Modal A10G GPUs. Costs are provider-metered
or conservative wall-clock estimates, as labelled in the artifacts.

## 7. Discussion

The null is informative because the interval is narrow at the top: any pooling advantage here is unlikely to
exceed about 3 points, and the point estimate slightly favours the simpler call. We see three reasons aggregation had
so little room to help.

The shared forecaster sets the answer. Every note is read by the same small model, 202 of 210 decisions were the same
under both rules, and the notes added only 2.4 points over no notes at all. Specialists built from one model seem to
contribute correlated information rather than independent information, and pooling of any kind, market-style or
not, cannot help much when that is true.

Retrieval and ordering matter more than aggregation: misses cost 18 points and candidate order flipped up to 23% of
picks.

What looked good on development data did not transfer. The development set suggested pooling improved calibration
and that the learned weights should favour the trace specialist. On held-out repositories the calibration advantage
reversed and the learned weights gave no accuracy gain. Three development repositories turn out to be a thin basis
for fitting even five weights.

The evidence-market framing is still a useful design vocabulary, but it only pays off when participants know
different things. Diverse models, executed tests or coverage data look like better sources of that diversity than five prompts
to one model.

## 8. Limitations

With only seven test repositories the cluster interval is wide. Mutants are not natural bugs, and single-function
mutants with templated issues favour trace evidence. One small model (Gemma 4 E4B) plays every role, so the
specialists share its blind spots. Order sensitivity is large, and averaging over orders is itself a design choice. Our "market" has no
incentives or trading; Hedge is a supervised pool. All development and held-out calls ran on Modal A10G GPUs in
bf16, and the batch a prompt lands in can change the low-order bits (identical prompts agreed on Top-1 99.5% of the
time across runs). One of the 1,050 held-out note generations (one specialist on mb-tomli-0010, a 27k-token prompt)
did not fit in A10G memory and was generated greedily on an L40S. Every probability-agent call ran on an A10G.

## 9. Conclusion

We built a market-inspired fault localizer in which specialists contribute evidence to a shared probability
forecast, and tested its central claim with a frozen, preregistered comparison on held-out repositories. On 210 bugs
from seven unseen repositories, the learned Hedge pool did not outperform a matched full-evidence baseline (−0.95
points, 95% repository-cluster interval [−4.3, +2.9]), and the two agreed on 96% of bugs. The limits came from
retrieval (18% of bugs never reached the candidate pool) and candidate ordering, not aggregation. An evidence market
for code needs participants with independent information; with one small model in every role, the market mostly
repeats what its forecaster already thinks.

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
