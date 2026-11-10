# BLOCKERS.md

## Human decisions (PLAN 15)
_2026-11-10_
1. MarketBugs repo list: Claude proposes, human approves (pending).
2. Gemma 4 variant: **DECIDED** `google/gemma-4-E4B-it@4ecc2ba6` (verified to exist on HF, not gated). Alternative `google/gemma-4-E2B-it@4adf1574`. Larger: 12B-it, 26B-A4B-it, 31B-it.
3. Kaggle Data tab: **DECIDED: none / skip.** Graph agent uses an `ast` call graph; Semantic agent uses an open embedding model.
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

## Session scope
- 2026-11-10: human asked for the work through the G1 pilot only, then a stop for review.

## Status 2026-11-10 (session 1)
- G0 GPU smoke PASSED: gemma-4-E4B-it on Modal A10G loads as Gemma4ForConditionalGeneration (transformers 5.22.2, torch 2.8.0+cu128); labels ' A'..' T' resolve as single tokens with the real chat template; token_mapping_hash 60e8077ec8db6a85; 3/3 probes top-1 correct, sums 1.0, rerun max diff 0.
- SWE-bench Lite verified: princeton-nlp/SWE-bench_Lite@6ec7bb89, test=300, dev=23. Sample of 100 (seed 20261110, proportional by repo): 79 single-function, 17 multi-function, 4 no-function (file-level bucket). 0 parse or checkout failures.
- **BLOCKER: running external repo test suites was denied by the agent's permission classifier.** MarketBugs generation (5.1), B3 Ochiai and E5 need it. Fallback (PLAN 12): the human runs `python -m src.gen.eval_repos`, then `python -m src.gen.build_marketbugs <repo>` locally, or grants permission.
- Deviation (VERIFY): mutant validity reruns (runs 2–3) re-run only the failing tests, not the full suite.
- VERIFY: vLLM cross-check not needed (an HF forward pass is used). Embedder BAAI/bge-small-en-v1.5 revision not pinned yet.
- Low disk: 5.5 GB free on the user's volume.
