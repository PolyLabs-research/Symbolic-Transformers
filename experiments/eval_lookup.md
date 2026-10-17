# Lookup diagnostic: results

**Status: complete. Neither world encoding taught either model to look atoms up.**

Setup and pre-registration: `plan_formula_evaluation.md`, section "Lookup diagnostic". Every formula is a
single atom, so an item asks only whether one atom holds in the world.

- **Runs.** Four, on an RTX 2000 Ada laptop GPU: `tiny` (2 layers) and `small` (4 layers, 3.5M
  parameters), each on both encodings. Settings: direct format, batch 64, learning rate 3e-4, 500 warm-up
  steps, 20 epochs, seed 1.
- **Data.** Regenerated locally, byte-identical to the committed metadata hashes.

Accuracy / minimal-pair accuracy (%):

| encoding | model | test_iid (2-4 objects) | test_large_world (5 objects, longer than training) |
|---|---|---|---|
| `true_atoms` (true atoms listed) | tiny | 64.3 / 29.3 | 54.1 / 9.8 |
| `true_atoms` | small | 50.0 / 0.0 | 50.0 / 0.0 |
| `full_table` (every atom with its value) | tiny | 58.2 / 16.4 | 55.5 / 11.0 |
| `full_table` | small | 58.3 / 16.6 | 53.4 / 6.8 |

Accuracy by number of objects, test splits pooled:

- `true_atoms` tiny: 66.3 / 66.7 / 60.3 / 54.1 for 2 / 3 / 4 / 5 objects.
- `full_table` tiny: 59.8 / 58.5 / 56.4 / 55.5.

## Against the pre-registration

- **Criterion: no run is learned.** Pair accuracy is at most 29.3%, against a threshold of 95%.
- **Predictions:**
  - "`full_table` is learned by `tiny`": false. It scores 16.4% pairs, below `true_atoms` (29.3%).
  - "`true_atoms` is not learned by `tiny` (pairs < 80%)": true.
  - `small` was left open. It failed both, and on `true_atoms` it never left a constant answer.
- **Decision rule.** Neither encoding is learned, so the rule reads: lookups need more capacity or data
  than these models have. It is not yet settled whether that is capacity, data, or optimisation. The
  next section explains why.

## Reading

**The copy encoding did not help, contrary to the prediction.** What makes a lookup hard here is
probably the key, not membership against copying.

- **An atom is spread over several tokens:**
  - `PRED p ( CONST a )`, where the key is the pair (p, a);
  - `PRED 4 ( CONST a , CONST b )`, a three-part key at different offsets.
- **What a lookup takes.** The model has to assemble that key at the query and at every listed atom,
  then match them.
- **Contrast with the FOL next-symbol model.** It learned to refer back to a bound variable, a
  single-token copy (`VAR 7` → `7`). Single-token copies are the case induction heads handle.
- **The full table is also twice as long,** so there are more places to search.

**Untested so far: training length, and how hard the key is.**

- The scratchpad model in Stage A trained 30 epochs, with several lookups supervised per item, and still
  sat at chance on lookups.
- Whether 20 epochs here stopped on a plateau, like the FOL model's epochs 2-5 before it learned to copy
  variables, needs the validation curves.
- The cheapest way to locate the bottleneck is a ladder of key complexity:
  1. single-token keys, a membership test among listed numerals;
  2. unary atoms only, a two-token key at fixed offsets;
  3. the full signature.

## Key-complexity ladder: results

The ladder is pre-registered in `plan_formula_evaluation.md`. It used the `true_atoms` encoding, `tiny`, 40
epochs, and the same settings otherwise, run on the RTX 2000 Ada GPU. Accuracy / minimal-pair accuracy on
test_iid (%):

| rung | the key | test_iid | accuracy by number of objects | validation loss over 40 epochs |
|---|---|---|---|---|
| `single` | one token | 60.4 / 22.5 | 66.6 / 57.4 / 56.2 (12 / 21 / 32 objects) | flat: 0.676 to 0.658 |
| `unary` | two tokens, fixed offsets | 59.1 / 18.1 | 61.9 / 59.1 / 56.4 (3 / 5 / 8) | rising: 0.678 to 0.710 |
| `full` | two or three tokens | 69.2 / 38.6 | 73.1 / 70.0 / 64.9 (2 / 3 / 4) | plateau to epoch 11, falls to 0.55 by epoch 17, 0.525 at 40 |

- **Criterion: no rung is learned.**
- **Predictions:** "`single` is learned" is false. "`full` is not learned" is true. `unary` was left open,
  and it was not learned either.
- **Decision rule.** `single` fails, so even a one-token membership test is not learned in this setup.
  The problem is upstream of the key structure, and a task known to be learnable has to be checked first.

**Reading.**

- **The ladder is not monotone in key complexity.** The rung with the hardest key did best, and it is
  also the only one whose loss broke through a plateau. So the key's structure is not the bottleneck.
- **Accuracy falls with the size of the world in every rung,** so how much has to be searched matters
  more than what is searched for.
- **`unary` fit its training items without generalising:** its validation loss rose across all 40
  epochs.
- **`full` broke through only after the learning rate had started its cosine decay,** so it may have been
  cut off partway.

The known-learnable task to check next is associative recall: key→value pairs, then queries. Small
transformers learn it reliably when trained with many queries per sequence. Every item here carries one
query and therefore one bit of supervision.

## Supervision-density check: results

This is the 2×2 on the `single` rung, pre-registered in `plan_formula_evaluation.md`: `tiny`, 40 epochs,
the same settings as before, on the GPU. Accuracy / minimal-pair accuracy on test_iid (%):

| | 1 query per training item | 8 queries per training item |
|---|---|---|
| `true_atoms` | 60.4 / 22.5 (ladder) | 55.6 / 18.3. Validation loss was flat at 0.69 until epoch 21, then fell to 0.652 |
| `full_table` (associative recall) | 59.3 / 18.6. Validation loss was flat at 0.665, then rose to 0.680 | **50.0 / 0.0. Validation loss was 0.693-0.695 for all 40 epochs** |

- **Criterion: none learned.**
- **Predictions:**
  - "`full_table` with 8 queries is learned": false. It never left a constant answer.
  - "`full_table` with 1 query is not learned": true.
  - `true_atoms` with 8 queries was left open. It was not learned, but started to move late.
- **Decision rule.** The textbook-learnable case fails in this pipeline, so the training setup itself has
  to be examined before anything else. `scripts/sanity_recall.py` trains the same model on minimal
  associative recall, freshly sampled every step, changing one setting at a time: post- vs
  pre-LayerNorm, dropout, and learning rate.

## Sanity check: the model can learn associative recall

`scripts/sanity_recall.py` trains `tiny` with the trainer's optimiser settings (AdamW, learning rate
3e-4, 500 warm-up steps, batch 64) on minimal associative recall: `k1 v1 ... k8 v8 ANSWER q -> v(q)`.
Keys are distinct numerals, and every batch is freshly sampled. Query accuracy is measured on fresh
sequences, with one query each.

| setup | step 1,000 | 3,000 | 4,000 | 5,000 | 6,000 |
|---|---|---|---|---|---|
| binary values, 1 query (default setup) | 0.61 | 0.65 | | | |
| the same, varied: pre-LN; no dropout; learning rate 1e-3; pre-LN with 1e-3 | 0.59-0.65 | 0.64-0.65 | | | |
| binary values, 8 queries | 0.65 | 0.64 | 0.71 | **1.00** | **1.00** |
| numeral values (100 classes), 1 query | 0.12 | 0.15 | 0.15 | 0.16 | 0.10 |
| numeral values, 8 queries | 0.13 | 0.16 | 0.16 | 0.16 | **0.99** |

**Reading:**

- **The model and training loop are not broken.**
- **Lookup is learned as a long plateau followed by a sudden jump.** With binary values, the plateau sits
  at about 64%, which is what answering with the majority of the listed values gives. The jump comes at
  4,000-6,000 steps with 8 queries per sequence. With 1 query it does not come within 6,000 steps.
- **Post- vs pre-LayerNorm, dropout and learning rate change nothing** within 3,000 steps.

**The open question is why the formula-evaluation lookups never made the jump.** The full-table run with
8 queries per item had 31K steps. Two differences from this check remain:

- **The layout.** The value comes three tokens after the key, sequences are about ten times longer, and
  the key also appears in the list of objects.
- **The data.** Those runs cycle through a fixed 50K items for 40 epochs, which lets training loss fall
  through memorisation instead. The 1-query full-table run did exactly that: its training loss fell
  while validation loss rose.

`--layout table` repeats this check in the exact full-table layout, on fresh data.

## Sanity check in the table layout: one filler token blocks the jump

The same check was run in four layouts, on the RTX 2000 Ada GPU. Setup: `tiny`, the default setup (post-LN,
dropout 0.1, learning rate 3e-4), 8 entries, binary values, 8 queries per training sequence, and seed 0.
Query accuracy is measured on 512 fresh one-query sequences.

| layout | sequence | steps | query accuracy |
|---|---|---|---|
| `pairs` | `k v ... ANSWER q -> v` | 10,000 | 0.60-0.66 to step 4,000, then **1.000** from step 5,000 (loss 0.003) |
| `pairs_offset2` | `k ) v ... ANSWER q ) -> v` | 10,000 | 0.61-0.66 throughout |
| `table_nodomain` | `SEP PRED 0 ( CONST k ) v ... MODELS PRED 0 ( CONST q ) ANSWER -> v` | 10,000 | 0.59-0.65 throughout |
| `table` | the same, after the list of objects `CONST k1 ... CONST kn` | 40,000 | 0.60-0.67 throughout |

On the GPU, `pairs` makes the jump at the same point as on the CPU (between steps 4,000 and 5,000). None of
the other three leaves the plateau.

**Reading:**

- **Fixed data is not the cause.** The table layout had 40,000 fresh steps, eight times what `pairs`
  needed, and never left the plateau. So the formula-evaluation lookups would have failed without
  memorisation.
- **The list of objects is not the cause either.** `table_nodomain` fails as well.
- **One filler token is enough to block learning.** `pairs_offset2` differs from `pairs` only by a `)`
  between each key and its value, and between the query key and the answer. It does not jump within
  10,000 steps, twice what `pairs` needed.
- **Why this is plausible.** `pairs` is the pattern an induction head handles (Olsson et al. 2022). The
  value comes right after its key, and the answer is predicted at the query key. The circuit has two
  parts:
  - A first-layer head copies each token's predecessor into it, so each value position carries its key.
  - A second-layer head at the query key attends to the position whose predecessor matches the current
    token, and copies that position's token.

  A filler breaks both parts:
  - In the facts, the first layer must look two tokens back.
  - In the query, the key is no longer the current token, so the match must also use something the first
    layer computed.

  The table layout has both breaks. Each value sits two tokens after its key, and the answer is predicted
  two tokens after the query key.
- **Limits.** Each layout ran with one seed, and a jump's timing varies between seeds. The result is
  "not within twice the steps `pairs` needed", not "never". A model with more layers might find a
  different circuit.

`pairs_offset2` changes the facts and the query together, so which of the two blocks the jump is still
open. The next check separates them, and tests a table layout with each value moved right after its key.
It is pre-registered in `plan_formula_evaluation.md`.
