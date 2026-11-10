# BLOCKERS.md

## Open human decisions (PLAN 15)
1. MarketBugs repo list: pending.
2. Gemma 4 variant: **provisional** `google/gemma-4-E4B-it@4ecc2ba6` (verified to exist on HF, not gated). Alternative `google/gemma-4-E2B-it@4adf1574`. Larger: 12B-it, 26B-A4B-it, 31B-it.
3. Kaggle Data tab: no competition or dataset identified in the repo; kaggle CLI not installed. Pending human.
4. E1 primary test: kept as defined, pending confirmation.
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
