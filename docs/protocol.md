# Frozen pilot protocol

## Research question

Does target-language domain generation followed by back-translation and mixed fine-tuning
adapt a small pretrained Arabic-English MT model? At matched synthetic-pool size and update
budget, does quality ranking help over random selection? What happens to generic-domain quality?

## Data

Read the official TICO English-Arabic archive. Reverse its fields for Arabic-to-English.
Group by the document portion of `stringID` (before `:`), shuffle document IDs with seed42,
reserve approximately 20% each for test/dev and the rest for training. Remove exact normalized
source OR target overlap in test/dev/train precedence, then sample 200 train, 100 dev, 200 test.
Keep the document partition manifest. This is not the official split or original paper protocol.
Only training English is prompted into the LM. OPUS-100 supplies 3600/100/200 generic rows
from train/validation/test, after exact exclusion against TICO and existing generic samples.
Near-copy filtering of synthetic text is a leakage guard, not a semantic decontamination guarantee.

## Generation and back-translation

120 prompts, 3 continuations per prompt, top-k50, top-p0.95, up to128 new tokens.
Use only new tokens, split complete sentences, language-check English, remove duplicates,
drop exact/near copies of seeds and held-outs. Back-translate with English-Arabic Marian,
beam5. Reject wrong script, duplicate, held-out near-copy, and overlength pairs.

## Quality ablation

Encode both languages using multilingual MiniLM. Pair fidelity proxy is cross-lingual cosine.
Domain proxy is cosine between generated English and centroid of authentic training English.
Score = 0.7 fidelity + 0.3 domain. Coefficients are not learned on test data.
K = min(400, floor(eligible pool * 0.5)); require K>=60 or fail visibly.
Random pool C and top-K pool D can overlap, intentionally: both come from the same eligible pool.
If K is small, treat the result as a feasibility pilot, not a strong reproduction.

## Training and evaluation

B/C/D each: independent initialization, 300 steps, batch4, accumulation4, lr2e-5,
max length128, validation/save every100, dev-loss checkpoint selection, 90% in-domain
sampling and 10% generic. Same sampled example count; actual token totals are reported.
E averages A/D compatible parameters with fixed alpha0.5. All systems decode with beam4.
Use unmodified predictions, corpus BLEU13a, chrF++ and TER on both domains. Record signatures.
Cluster bootstrap compares D-C chrF++ by test document; few clusters and one seed limit inference.
Do not use intervals from this pilot as a substitute for independent multi-seed experiments.

## Interpretation

- C vs A: combined synthetic adaptation effect, not isolated benefit of the LM.
- C vs B: comparison with small authentic adaptation, confounded by different domain pool sizes.
- D vs C: matched-size ranking ablation (heuristic filtering rather than guaranteed correction).
- E vs D: fixed checkpoint-averaging ablation.
- Generic scores reveal potential catastrophic forgetting.

Further causal isolation would need a target-monolingual back-translation control without LM
generation, additional seeds and matched token budgets. These are future work, not completed claims.
