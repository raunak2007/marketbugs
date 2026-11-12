# ANALYSIS_PLAN.md (preregistration; committed before any test-split result exists)

Written from PLAN.md section 10 on 2026-11-10. Frozen with the rest of the protocol at G2 (`configs/FROZEN.txt`).

## Primary (confirmatory) test: the only one
- **Comparison:** Top-1 accuracy of **P_hedge** (Hedge log pool of the five C-mode singleton posteriors `{i}`; weights learned on dev and frozen; label: *learned supervised pool*) vs **B_full5** (mean of 5 cyclic note-order permutations of the full-evidence probability-agent call).
- **Data:** MarketBugs **test** split, `test_informed` condition, all instances (unconditional on gold-in-pool).
- **Matched inputs:** both methods use the same five notes and five probability-agent calls; note-generation cost counted for both.
- **Statistic:** paired per-instance difference in Top-1 (P_hedge minus B_full5), averaged.
- **Inference:** repo-cluster bootstrap, two-stage (resample repos with replacement, then instances within each sampled repo with replacement), B = 10,000, seed 20261110. Report 95% percentile CI. The instance-level bootstrap CI is reported **alongside**, never instead.
- **Per-repo table** of both methods and the difference is mandatory.
- With roughly 7 test repos the cluster interval will be wide. An inconclusive result is reported as inconclusive. We do not switch to instance-level inference.

## Fixed protocol constants
- Order-sensitivity threshold: if more than **15%** of dev top-1 predictions change across 3 candidate-order permutations, every method averages log-probs over 2 fixed permutations (a chosen value, not derived).
- Gold function minimum: 3 statements. Dev cap 40/repo (at most 120); test cap 30/repo (at most 210).
- Pool K = 20. K = 10 vs 20 comparison is reported, not acted on.
- Temperature: one scalar per method, fit on dev by NLL. Hedge eta chosen from {0.1, 0.3, 1, 3} by 2-fold CV on dev.

## Exploratory (effect sizes and CIs only, no significance claims)
Top-3, Top-5, MRR, NLL, Brier, ECE (top-label, 10 equal-width bins); by stratum (direct, near, far); `no_trace` condition; gold-in-pool subset vs all; SWE-bench Lite transfer (descriptive, possible pretraining exposure); F-mode pools; all baselines B0–B5, P_vote, P_avg, P_geo, P_stack; E2 evidence valuation; E3 redundancy; E4 sequential market; E5 repair.

## Reporting rules
- Cluster and instance-level CIs side by side.
- "Gold in pool" and unconditional results everywhere.
- Observed strata distribution reported; no rebalancing.
- Every exclusion reported with count and reason.
- Per-repo breakdown for every headline result.
- Total cost per method (note generation + all probability-agent calls).
- Null and negative results get the same tables as positive ones.
- Credit labels: Shapley/marginal/pivotal credit = "value of an evidence source given the fixed probability agent".
