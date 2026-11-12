# MarketBugs G3 bundle (held-out result + paper)

- `MARKETBUGS_PAPER.md` — paper draft (see word count in the delivery note).
- `results/` — held-out evaluation over all 210 test bugs: `RESULTS.md` (tables), `summary.json`, `per_instance.csv`
  (one row per bug; every headline number reproduces from it), `per_repo.csv`, `confidence_intervals.json`
  (two-stage repository-cluster bootstrap, 10,000 replicates, seed 20261110).
- `frozen/` — the frozen evaluation contract (`FROZEN.txt`) and dev-fit parameters (eta, Hedge weights, temperatures).
- `protocol/` — preregistration (`ANALYSIS_PLAN.md`), experiment plan, decision log (`BLOCKERS.md`, incl. G2/G3).
- `dev/` — G2 development report (exploratory; same data used for fitting).
- `run/` — held-out run manifest and completion status.
- `SHA256SUMS` — checksums of every file here.

Headline: P_hedge 134/210 (63.8%) vs B_full5 136/210 (64.8%); paired difference −0.95 pp, 95% repo-cluster CI
[−4.3, +2.9] (inconclusive). Pool recall 172/210. Code and full artifacts: https://github.com/raunak2007/marketbugs
