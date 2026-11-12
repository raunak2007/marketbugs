# MarketBugs: What Makes Evidence Aggregation Reliable for Small-Model Software Debugging?

Code, benchmark, frozen parameters and per-bug results: https://github.com/raunak2007/marketbugs. Every number in
this paper is read from `artifacts/validation/heldout_v1/` (held-out) or `artifacts/g2_dev_report.json`
(development) in that repository.

## Abstract

Debugging agents must localize a fault from partial, uncertain signals before they can repair it. We study how such
evidence should be aggregated, using a market-inspired framework in which five specialists contribute notes,
candidate functions act as competing hypotheses, and a frozen Gemma 4 E4B probability agent turns any coalition of
notes into an explicit forecast over 20 retrieved candidates. We release MarketBugs, 330 validated single-function
mutants from ten Python repositories with repository-disjoint development and test splits, and run a preregistered
evaluation on 210 bugs from seven unseen repositories. The full-evidence baseline (Full5) localized 136/210 bugs at
Top-1 (64.8%), or 79.1% of the 172 bugs whose faulty function was retrieved. Retrieval recall was 81.9% (172/210),
which caps every downstream method. Specialist notes raised Top-1 from 131/210 to 136/210 (+2.4 points), but the
95% repository-cluster interval (−3.3 to +8.1) includes zero. The preregistered primary test found no demonstrated
advantage for a learned Hedge pool over Full5: 134/210 versus 136/210, a paired difference of −0.95 points (95%
interval −4.3 to +2.9). Single-order predictions changed under candidate reordering on 11% to 23% of bugs. Reliable
aggregation appears to depend on the availability, relevance and stable interpretation of evidence more than on
the pooling rule.

## 1. Introduction

Repair agents localize first and edit second, so a wrong localization wastes every later step. The hard part is
less collecting diagnostic signals than knowing which help and how to combine them reliably.

In a prediction market, participants with different information move a shared forecast until it reflects what they
collectively know. We borrow the structure, not the economics. In our evidence
market, each specialist is an information contributor, each candidate function is a competing hypothesis about the
fault, and a probability distribution over candidates is the collective belief. Coalition evaluation, pooling and
sequential updating then correspond to partial participation, combining forecasts, and contributions in turn. We use "evidence market" in the operational
sense of aggregating specialized information into a shared probability forecast, rather than claiming an
incentive-compatible financial prediction market: nothing in the system trades, clears a price, or rewards
truthful reporting.

Our question is: what makes evidence aggregation useful and reliable for small-model debugging agents, and can an
evidence-market framework expose the bottlenecks that limit it? We make three contributions:

1. **An evidence-market framework** that turns specialist diagnostic notes into explicit probability distributions
   over candidate functions and supports coalition evaluation, probability pooling and sequential belief updating.
2. **MarketBugs**, a reproducible benchmark of 330 validated single-function mutants across ten Python repositories,
   with repository-disjoint splits and evaluator-only gold labels.
3. **A preregistered held-out study** (our primary empirical result) that measures a retrieval ceiling,
   candidate-order instability and a modest observed benefit from evidence, and finds no demonstrated advantage for
   learned pooling over a strong matched baseline.

## 2. Related Work

Spectrum-based fault localization ranks code by how often failing and passing tests execute it, as in Tarantula
(Jones & Harrold 2005) and Ochiai (Abreu et al. 2007); Wong et al. (2016) survey the field. Defects4J (Just et al.
2014) established reproducible fault benchmarks with failing tests. LLM-based software engineering is often
evaluated on SWE-bench (Jimenez et al. 2024), and many systems localize through staged retrieval and ranking before
repairing, for example Agentless (Xia et al. 2024) and AutoFL (Kang et al. 2024). We study localization alone,
with no repair step.

Forecast aggregation combines probabilistic opinions through linear or logarithmic pools (Genest & Zidek 1986), and
the Hedge algorithm weights experts by exponentiated cumulative performance (Freund & Schapire 1997). Prediction
markets aggregate dispersed information into prices (Wolfers & Zitzewitz 2004), and market scoring rules reward a
participant by the change in log score it causes (Hanson 2003). Our sequential mode borrows that scoring without
any incentive claim. Shapley values (Shapley 1953) attribute a coalition's value to its members; our coalition
tables measure the value of an evidence source given a fixed probability agent, not a causal property of the bug.

## 3. Methods: an evidence market over candidate functions

**Candidate retrieval.** For each bug we build a pool of K = 20 functions by interleaving BM25, embedding similarity
and call-graph neighbours of traceback frames, then shuffle them with a seed that does not encode retrieval rank.
The gold function is never inserted, so a retrieval miss is a failure for every method.

**Evidence generation.** Five specialists each read a different view and write a ranked note (temperature 0, at
most 150 tokens): trace (the failing test and its frames), graph (call and import neighbours), spec (docstrings and
type hints against the code), static (AST risk patterns) and semantic (embedding similarity to the issue).

**The shared forecast.** A frozen probability agent, Gemma 4 E4B-it at a pinned revision in bf16, reads the issue,
the candidates and any subset of notes, and stops at an answer cue. A softmax over the logits of the 20
single-token candidate labels gives a distribution over hypotheses. Running this for note subsets gives coalition
values v(S) = log p_S(gold) − log p_∅(gold), which we call evidence-coalition valuation (Mode C).

**Aggregation.** Full5 is one call with all five notes, averaged over the five cyclic note orders. P_hedge, a
supervised probability pool, combines the five single-note forecasts in a weighted logarithmic pool with weights
w_i ∝ exp(η G_i) from cumulative development log scores. Both use five notes and five probability-agent calls per
candidate order, so the comparison isolates how evidence is combined. Mode S is market-inspired sequential belief
updating: contributors revise a shared forecast one at a time and are scored by the change in log score they cause.
Mode S was run on development data only.

**Order averaging.** Because Top-1 picks on development data changed with candidate display order for 25.7% (30-bug
pilot) and 33% (all 120 bugs) of predictions, above our preregistered 15% threshold, every method averages
log-probabilities over two fixed candidate permutations, aligned by candidate ID.

## 4. The MarketBugs benchmark

We screened 16 Python libraries for permissive licenses, pure-Python code and fast, deterministic suites; ten
passed. A seeded split fixed before any results assigned three repositories to development (cachetools,
intervaltree, sortedcontainers; 120 bugs) and seven to test (attrs, boltons, dateutil, isodate, more-itertools,
tomli, toolz; 210 bugs, 30 each).

Each bug is a minimal AST edit to one function (comparison flip, operator swap, off-by-one, argument swap, removed
condition, wrong variable, deleted statement or changed return). A mutant is kept only if the original suite passes,
the mutant fails at least one test identically over three runs, exactly one function changed, and that function has
at least three statements. Issue text is generated from the failing test. Gold locations, patches and operators sit
in evaluator-only fields; inference code reads instances through a whitelisted view, and leakage tests check every
prompt builder. Bugs are stratified by failure visibility: 200 direct (gold in the traceback), 41 near (one call
away) and 89 far.

Known gold locations and deterministic failures let MarketBugs separate retrieval from ranking for controlled studies
of evidence use. It is synthetic, and results on it do not
establish performance on naturally occurring bugs.

## 5. Experiments

**Frozen protocol.** Before any test inference we froze, in `configs/FROZEN.txt` and `artifacts/frozen_params.json`
(code commit `bda0716`), the model revision, prompts, K = 20, the two-permutation protocol, η = 0.1 (2-fold
cross-validation over {0.1, 0.3, 1, 3}), the Hedge weights (trace .380, semantic .236, spec .149, static .123, graph
.113) and per-method temperatures, all fit on the 120 development bugs. The runner refuses to start if code,
prompts, configs or the split changed after the freeze, and gold is attached only by a separate evaluator after
inference finishes.

**Primary (confirmatory) comparison.** The single preregistered test is the mean paired Top-1 difference P_hedge
minus Full5 over all 210 test bugs, with retrieval misses counted as wrong. Uncertainty comes from a 10,000-replicate
two-stage repository-cluster bootstrap (resample the seven repositories, then bugs within each sampled copy; seed
20261110; percentile 95% interval). An interval spanning zero is inconclusive. All other analyses are exploratory.

**Metrics.** Top-1/3/5, mean reciprocal rank, pool recall, conditional Top-1 given gold in pool, and tempered
log-loss over gold-in-pool bugs (retrieval misses have no finite log score).

**Disclosed changes.** Before the freeze, issue text was capped at 4,096 tokens (first 1,024 and last 3,072), which
altered exactly two development bugs with runaway doctest output. To fit compute, test inference ran only the calls
these analyses need (22 probability-agent calls and 5 note generations per bug). Mode S, a placebo-note control, the
full 32-subset coalition table, conventional baselines such as Ochiai, and SWE-bench transfer were not run on the
test set.

## 6. Results

### 6.1 Held-out localization and the retrieval ceiling

| Method | Top-1, all 210 | Top-1, gold in pool (172) | Top-3 | MRR |
|---|---|---|---|---|
| No notes | 131 (62.4%) | 76.2% | 74.8% | .691 |
| All notes, one order | 136 (64.8%) | 79.1% | 75.7% | .706 |
| Full5 (matched baseline) | 136 (64.8%) | 79.1% | 75.7% | .707 |
| P_hedge (supervised pool) | 134 (63.8%) | 77.9% | 74.8% | .702 |

On seven unseen repositories Full5 ranked the faulty function first for 136 of 210 bugs, against a within-pool
chance rate of 1 in 20. We did not evaluate conventional fault-localization baselines here, so we make no
comparative claim against them.

The faulty function reached the candidate pool for 172 of 210 bugs (81.9%). For the other 38, downstream
forecasting cannot select the correct function, so 81.9% is the highest unconditional Top-1 any method can reach
with these pools. Given a successful retrieval, Full5 ranked the faulty function first 79.1% of the time. Retrieval
misses therefore cost about 18 points, more than the full spread between methods. Better retrieval would not
automatically recover all 38, since new candidates must still be ranked correctly. Accuracy also tracked
failure visibility: on direct bugs (n = 118) both primary methods were about 83% correct, and on near (n = 28) and
far (n = 64) bugs about 40%.

### 6.2 The role of specialist notes (exploratory)

With no notes the probability agent localized 131/210 bugs; with all five notes in one call it localized 136/210.
The paired difference is +2.4 points (11 wins, 6 losses, 193 ties) with a cluster interval of [−3.3, +8.1]. The
evidence produced a modest positive observed difference, but the experiment did not establish a reliable
improvement across repositories. Per-repository effects ranged from −6.7 points (attrs) to +10.0 points (boltons and
toolz).

### 6.3 Preregistered primary comparison: P_hedge versus Full5

P_hedge localized 134/210 bugs and Full5 136/210. The paired difference is −0.95 percentage points, with a 95%
two-stage repository-cluster bootstrap interval of [−4.3, +2.9]; a diagnostic instance-level bootstrap gives
[−3.8, +1.4]. The result is inconclusive: it does not establish a P_hedge advantage, and it does not show the methods
are equivalent. Correctness differed on 8 bugs (P_hedge right on 3, Full5 right on 5), with 202 correctness ties.
Measured separately, the two methods selected the same function on 193/210 bugs (91.9%). Of the 17 bugs where they
chose differently, 9 were missed by both.

| Repository | n | P_hedge | Full5 | Difference |
|---|---|---|---|---|
| attrs | 30 | .367 | .367 | 0 |
| boltons | 30 | .633 | .667 | −.033 |
| dateutil | 30 | .633 | .667 | −.033 |
| isodate | 30 | .633 | .633 | 0 |
| more-itertools | 30 | .700 | .733 | −.033 |
| tomli | 30 | .833 | .767 | +.067 |
| toolz | 30 | .667 | .700 | −.033 |

Every per-repository difference is one or two bugs. Difficulty varied far more across repositories (Full5 from .367
on attrs to .767 on tomli) than across methods.

### 6.4 Candidate-order instability

We measured, for each note subset, how often the single-order Top-1 prediction under candidate permutation 0
differed from that under permutation 1, before averaging. The faulty function and the evidence are the same in both
orders; only the display order changes. The prediction changed on 48/210 bugs (22.9%) with no notes, 29 to 41 bugs
(13.8% to 19.5%) with one note, and 24/210 (11.4%) with all five notes. We did not measure the stability of the
averaged predictions, so we do not claim that averaging removed this instability.

### 6.5 Development-to-test probability quality

On development data, P_hedge had the best tempered log-loss (.660, against .697 for Full5 and .723 for the
single-order all-notes call) and was nearly identical to an equal-weight pool (.663), with η at the lowest grid
value. On the held-out gold-in-pool bugs the ordering reversed: Full5 .749, P_hedge .814 (Brier score over all 210
bugs: .346 and .382). This is a development-to-test generalization failure for these learned weights, not evidence
that learned aggregation is inferior in general.

### 6.6 Cost

Both primary methods use five note generations and ten probability-agent calls per bug, an estimated $0.020 per bug,
of which $0.015 is the shared notes. The held-out run metered about $7.5 on Modal A10G GPUs.

## 7. Discussion

Three findings are directly observed. Retrieval capped accuracy at 81.9%, and its misses cost more than any
difference between aggregation methods. Single-order predictions changed under candidate reordering on 11% to 23% of
bugs. And neither specialist evidence (+2.4 points) nor learned pooling (−0.95 points) produced a difference whose
interval excludes zero.

Possible explanations remain untested. Specialists built from one small model may contribute correlated rather than
independent information, and pooling helps most when contributors know different things. Another is that the probability agent's reading of
a note depends on presentation, which the order results are consistent with. The decline in instability as notes
are added (22.9% with none, 11.4% with all five) suggests evidence may also stabilize interpretation, an
observational pattern we did not test causally.

For autonomous debugging agents, this is actionable. An agent working through ranked functions should track whether
the faulty function was ever a candidate, since no reasoning recovers a missed retrieval. Rankings that change with arbitrary presentation order should be treated as uncertain, either by
averaging over orders or by re-checking before an edit. And any new aggregation mechanism should be judged against
a matched baseline with the same evidence and call budget. Our learned pool looked better on three development
repositories but not on seven new ones.

Because every coalition yields an explicit distribution over the same hypotheses, the framework evaluates
retrieval, interpretation and aggregation separately, which an end-to-end repair score would hide.

## 8. Limitations

Seven test repositories give only seven bootstrap clusters, and a 95% interval is not a guaranteed bound on
generalization. MarketBugs is synthetic, and templated issues with tracebacks favour trace evidence. One small model
plays every role. Mode S, placebo notes, the full coalition table, conventional baselines and SWE-bench transfer
were not evaluated on the test set. In bf16, identical prompts agreed on Top-1 99.5% of the time across runs, and one of
1,050 held-out notes was generated on an L40S after exceeding A10G memory.

## 9. Conclusion

MarketBugs contributes a reproducible evidence-market framework, a benchmark of 330 validated mutants, and a
preregistered held-out evaluation. On 210 bugs from seven unseen repositories, the framework localized 136 faults
at Top-1 (79.1% when the faulty function was retrieved). The experiments locate the main constraints in retrieving
the right code and keeping predictions stable under presentation changes, while learned pooling showed no
demonstrated advantage over a strong matched baseline. Whether evidence markets can outperform simpler systems
remains unestablished; the framework and benchmark give future debugging agents a way to test it.

## References

- Abreu, R., Zoeteweij, P., van Gemund, A. J. C. (2007). On the accuracy of spectrum-based fault localization. TAICPART-MUTATION 2007, pp. 89-98. doi:10.1109/TAIC.PART.2007.13
- Freund, Y., Schapire, R. E. (1997). A decision-theoretic generalization of on-line learning and an application to boosting. Journal of Computer and System Sciences 55(1).
- Genest, C., Zidek, J. V. (1986). Combining probability distributions: a critique and an annotated bibliography. Statistical Science 1(1), pp. 114-135. doi:10.1214/ss/1177013825
- Hanson, R. (2003). Combinatorial information market design. Information Systems Frontiers 5(1), pp. 107-119.
- Jimenez, C. E. et al. (2024). SWE-bench: Can language models resolve real-world GitHub issues? ICLR 2024.
- Jones, J. A., Harrold, M. J. (2005). Empirical evaluation of the Tarantula automatic fault-localization technique. ASE 2005, pp. 273-282. doi:10.1145/1101908.1101949
- Just, R., Jalali, D., Ernst, M. D. (2014). Defects4J: a database of existing faults to enable controlled testing studies for Java programs. ISSTA 2014.
- Kang, S., An, G., Yoo, S. (2024). A quantitative and qualitative evaluation of LLM-based explainable fault localization. FSE 2024. doi:10.1145/3660771
- Shapley, L. S. (1953). A value for n-person games. Contributions to the Theory of Games II.
- Wolfers, J., Zitzewitz, E. (2004). Prediction markets. Journal of Economic Perspectives 18(2).
- Wong, W. E., Gao, R., Li, Y., Abreu, R., Wotawa, F. (2016). A survey on software fault localization. IEEE Transactions on Software Engineering 42(8), pp. 707-740. doi:10.1109/TSE.2016.2521368
- Xia, C. S., Deng, Y., Dunn, S., Zhang, L. (2024). Agentless: demystifying LLM-based software engineering agents. arXiv:2407.01489.
