# MarketBugs held-out results (protocol v1.0)

n = 210 instances; gold in the 20-candidate pool for 172 (38 retrieval misses scored as failures in all unconditional metrics).

## Top-1 / Top-3 / Top-5 / MRR

| method | all: top1 | top3 | top5 | MRR | gold-in-pool: top1 | NLL (T, gold-in-pool, n) |
|---|---|---|---|---|---|---|
| p_empty | 0.6238 | 0.7476 | 0.7667 | 0.6907 | 0.7616 | 0.9281 (172) |
| p_full | 0.6476 | 0.7571 | 0.7714 | 0.7059 | 0.7907 | 0.785 (172) |
| B_full5 | 0.6476 | 0.7571 | 0.7667 | 0.7066 | 0.7907 | 0.7494 (172) |
| P_hedge | 0.6381 | 0.7476 | 0.781 | 0.7017 | 0.7791 | 0.8143 (172) |
| single.trace | 0.6143 | 0.7714 | 0.781 | 0.6915 | 0.75 | 0.8707 (172) |
| single.graph | 0.6381 | 0.7619 | 0.781 | 0.703 | 0.7791 | 0.8209 (172) |
| single.spec | 0.6381 | 0.7476 | 0.7667 | 0.6998 | 0.7791 | 0.8694 (172) |
| single.static | 0.6619 | 0.7571 | 0.7667 | 0.715 | 0.8081 | 0.7728 (172) |
| single.semantic | 0.6333 | 0.7524 | 0.7714 | 0.6977 | 0.7733 | 0.8667 (172) |

NLL is over gold-in-pool instances only (retrieval misses have no finite log score; count excluded is stated per method in summary.json). ECE over 210 cases / 7 repositories is a noisy estimate.

## Paired Top-1 differences

| contrast | status | observed diff | W / L / T | n |
|---|---|---|---|---|
| P_hedge − B_full5 | confirmatory (preregistered) | -0.0095 | 3 / 5 / 202 | 210 |
| p_full − p_empty | exploratory | 0.0238 | 11 / 6 / 193 | 210 |
| real_vs_placebo | omitted | p_full or placebo_full not available | | |

<!-- BOOTSTRAP:BEGIN -->
## Paired Top-1 differences with 95% repository-cluster bootstrap CIs

10000 replicates, seed 20261110. Instance-level CI is diagnostic only.

| contrast | status | observed | cluster 95% CI | instance CI (diag.) | verdict |
|---|---|---|---|---|---|
| P_hedge - B_full5 (Top-1) | confirmatory (preregistered) | -0.0095 | [-0.0429, +0.0286] | [-0.0381, +0.0143] | inconclusive (CI spans 0) |
| p_full - p_empty (Top-1) | exploratory | +0.0238 | [-0.0333, +0.0810] | [-0.0143, +0.0619] | inconclusive (CI spans 0); exploratory |
| real_vs_placebo | omitted | column diff.real_vs_placebo not present | | | |
<!-- BOOTSTRAP:END -->

## Ordering instability (not accuracy)

- cand_perm_top1_disagree.mask0: 0.2286 (n = 210)
- cand_perm_top1_disagree.mask1: 0.1667 (n = 210)
- cand_perm_top1_disagree.mask2: 0.1905 (n = 210)
- cand_perm_top1_disagree.mask4: 0.1381 (n = 210)
- cand_perm_top1_disagree.mask8: 0.1952 (n = 210)
- cand_perm_top1_disagree.mask16: 0.1714 (n = 210)
- cand_perm_top1_disagree.mask31: 0.1143 (n = 210)
- cand_perm_top1_disagree.placebo_full: None (n = 0)

## Cost

```
{
 "note_generation_usd_per_instance": 0.01467,
 "note_calls_per_instance": 5.01,
 "label_calls_per_instance_by_method": {
  "p_empty": 2,
  "p_full": 2,
  "B_full5": 10,
  "P_hedge": 10,
  "placebo_full": 2,
  "S_one_order": 10
 },
 "usd_per_instance_by_method_incl_amortized_notes": {
  "p_empty": 0.00113,
  "p_full": 0.0158,
  "B_full5": 0.02032,
  "P_hedge": 0.02032,
  "placebo_full": 0.0,
  "S_one_order": 0.01467
 },
 "placebo_note": "placebo reuses donor notes already generated for the real arm (no new note cost)",
 "total_est_usd_logged": 5.6915
}
```

## Provenance

```
{
 "head": "a9e784f73605e2f5845604ae5b489197b6fc4f29",
 "frozen_code_commit": "bda071662d1aea379990ddd0b170a2459b39169b",
 "frozen_params_sha256": "75709ef4033eb7f61c286add873faef2ac638161aca3d8ccb3ac6a7e94dd8959",
 "code_changed_since_freeze": [],
 "model": {
  "dtype": "bfloat16",
  "hf_id": "google/gemma-4-E4B-it",
  "revision": "4ecc2ba63bcac23ba4a767fc87025b1450f84c9c",
  "revision_full": "google/gemma-4-E4B-it@4ecc2ba63bcac23ba4a767fc87025b1450f84c9c",
  "token_mapping_hash": "60e8077ec8db6a85"
 },
 "protocol": "avg2",
 "seeds": {
  "placebo": 20261111,
  "bootstrap": 20261110,
  "generation": 0
 },
 "hedge": {
  "eta": 0.1,
  "weights": {
   "graph": 0.11323042421073878,
   "semantic": 0.23566029016938708,
   "spec": 0.14871292636837413,
   "static": 0.1228562049296381,
   "trace": 0.3795401543218619
  }
 },
 "evaluated_utc": "2026-11-12T21:23:00Z"
}
```
