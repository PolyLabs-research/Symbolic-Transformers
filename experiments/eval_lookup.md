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

## Where the filler blocks the jump: results

Pre-registered in `plan_formula_evaluation.md`. Same setup as the previous section.

| layout | steps | query accuracy |
|---|---|---|
| `table_keylast` (`... PRED 0 ( CONST k v ) ... MODELS PRED 0 ( CONST q -> v )`) | 20,000 | 0.65-0.70 to step 6,000, then **1.000** from step 8,000 (loss 0.003) |
| `pairs_gap_context` (`k ) v ... ANSWER q -> v`) | 10,000 | 0.60-0.65 to step 5,000, 0.994 at 6,000, then **1.000** |
| `pairs_gap_query` (`k v ... ANSWER q ) -> v`) | 10,000 | 0.60-0.66 through step 6,000; the run is still going |

- **`table_keylast` is learned, as predicted.** It is the same table layout that sat on the plateau for
  40,000 steps, with two changes: each value moved right after its key, and the answer predicted at the
  query key. With those changes it jumps between steps 6,000 and 8,000. `pairs` jumped at 5,000, and these
  sequences are more than three times longer.
- **Decision rule.** `table_keylast` is learned, so the lookup data gets a world encoding with each value
  right after its atom's last key token (`--encoding keylast`). The 8-query ladder then runs on it.
- **A gap in the facts alone does not block the jump, as predicted.** `pairs_gap_context` learns, one
  thousand steps after `pairs`. When the answer is predicted at the query key, a value two tokens after
  its key is still found. Whether the gap in the query alone blocks it awaits the end of
  `pairs_gap_query`.

## One token per symbol, on the key-last ladder: results

This is the 7-run ladder pre-registered in `plan_formula_evaluation.md`. Setup: `tiny`, 40 epochs, 8
queries per training item, on the RTX 2000 Ada GPU. Columns: test_iid accuracy / minimal-pair accuracy
(%), and the validation loss over the 40 epochs.

| # | rung | encoding | tokens | test_iid | validation loss |
|---|---|---|---|---|---|
| 1 | `single` | key-last | split | 50.0 / 0.0 | flat, 0.693-0.697 |
| 2 | `single` | key-last | fused | 59.9 / 19.8 (by objects: 63.1 / 59.4 / 57.4) | rising to 0.702 by epoch 35, then 0.696, 0.693, 0.688, 0.675, 0.656 in epochs 36-40 |
| 3 | `single` | table | fused | 50.0 / 0.0 | flat, 0.693-0.696 |
| 4 | `unary` | key-last | split | 50.0 / 0.0 | rising, 0.693 to 0.710 |
| 5 | `unary` | key-last | fused | 50.0 / 0.0 | rising, 0.693 to 0.714 |
| 6 | `full` | key-last | split | 50.0 / 0.0 (test_large_world 50.0 / 0.0) | |
| 7 | `full` | key-last | fused | 50.0 / 0.3 (test_large_world 50.2 / 0.6) | rising, 0.693 to 0.701 |

**Against the pre-registration:**

- **Predictions:**
  - "Runs 1 and 2 are learned": false. Run 1 never left chance. Run 2 started to fall in its last 5
    epochs, when the learning rate was at its 10% floor, and ended at 19.8% pairs.
  - "Run 3 is not learned": true.
  - Runs 4 and 5 were left open. Neither was learned, and "fused and split within 10 points" holds
    (0 and 0).
  - "Runs 6 and 7 are not learned": true.
- **Decision rule:**
  - The first clause needed runs 1 and 2 to learn, and they did not.
  - "Fused beats split by 10 or more points on a rung" is met to the letter on `single` (19.8 against
    0.0). The evidence is one seed, and a jump caught as it began. The timing of a jump varies between
    seeds, so this is not acted on until the factor check below repeats the comparison on fresh data.
  - The `unary` clause assumed `single` would be learned. It was not, so the obstacle comes before
    multi-part keys.

**Reading:**

- **The key-last layout that learned in the recall check does not learn here, even with one-token keys.**
  Three things differ between the two setups:
  - **Table size.** These tables hold 12-32 facts; the check's held 8.
  - **Data.** Here the model cycles through a fixed set of 50K items for 40 epochs. The check drew a
    fresh batch every step.
  - **Learning rate.** The trainer decays it with a cosine to 10%, and holds it there for the last 20% of
    training. The check kept it constant after warm-up.
- **The fixed data shows in the curves.** Validation loss rises clearly above chance in runs 4, 5 and 7
  (to 0.70-0.71), and in run 2 before its late fall. Training loss dips below chance at the same time
  (run 7: training 0.681, validation 0.700 at epoch 40). Those models fit the training items some other
  way instead of learning the lookup. Runs 1 and 3 stay within 0.004 of chance throughout.
- **Next.** The factor check in `plan_formula_evaluation.md` adds each difference, one at a time, to the
  layout that learned.

## Factor check: results

This is the check pre-registered in `plan_formula_evaluation.md`, run on the RTX 2000 Ada GPU. Each run
starts from `table_keylast` (8 facts, fresh batches, constant learning rate), which jumped between steps
6,000 and 8,000. Query accuracy is measured on fresh sequences.

| run | change | query accuracy |
|---|---|---|
| A | 21 facts | 0.56-0.64 for all 20,000 steps: no jump |
| B | fixed pool of 50,000 sequences | 0.641 at step 2,000, **1.000** from step 4,000 (training loss 0.000 from step 8,000) |
| C | cosine schedule | 0.645 at step 6,000, **1.000** from step 8,000 |
| D | fused tokens | 0.660 at step 8,000, 0.973 at 10,000, **1.000** from 12,000 |
| E | 21 facts, fixed pool, cosine, 31,000 steps | 0.57-0.66 to step 12,400, 0.996 at 15,500, **1.000** from 18,600 |

Reading:

- **Table size delays the jump.** On fresh data at 21 facts, about the middle of the `single` rung's
  12-32, no jump comes in 20,000 steps; with 8 facts it came by step 8,000. The prediction ("learned,
  later than 8,000") was wrong within this budget. Run E below jumps at 21 facts by step 15,500, so the
  delay is not a block.
- **Fixed data does not block it.** Run B jumped earlier than fresh data (by step 4,000). A training loss
  of 0.000 is the expected optimum here: every answer is determined by its sequence. Accuracy on fresh
  sequences is 1.000, so the model is doing the lookup rather than memorising. The rising validation
  loss in the ladder runs is what happens while a model sits on a long plateau. It is not, by itself,
  why the jump never came.
- **The cosine schedule does not matter.** Run C jumps at the same point as the constant rate.
- **Fusion delays the jump slightly.** Run D jumps about 2,000-4,000 steps later than split tokens. That
  is one seed each, and the ladder's only movement was in a fused run, so there is no consistent effect
  either way.
- **Run E learns, although it has the ladder's conditions.** 21 facts, a fixed pool of 50,000 and the
  cosine schedule over 31,000 steps give a jump at step 15,500. That is earlier than run A managed with
  fresh data, which matches run B: a fixed pool speeds the jump rather than slowing it.
- **Against the pre-registration:**
  - A, "learned, later than 8,000": false. There was no jump in 20,000 steps.
  - B was left open: learned.
  - C and D, "learned": true.
  - E, "not learned (reproduces the ladder)": false.
  - The decision rule for E learning applies: this script does not reproduce the ladder. The difference
    lies in the trainer or in the generated items.
- **Differences that remain between run E and the ladder's `single` run:**
  - **The trainer.** AdamW with betas (0.9, 0.98) and eps 1e-9, against (0.9, 0.999) and 1e-8 here;
    length-bucketed batches; seed 1 against 0.
  - **Table size.** The rung mixes 12, 21 and 32 facts; run E has 21 every time.
  - **Balanced queries.**
    - The ladder's training queries are TRUE half the time, whatever the table holds. Here a query picks
      a fact uniformly, so answering with the majority of the listed values scores about 0.6 on the
      plateau. Every sanity run sat at 0.60-0.67 before its jump. With balanced queries that shortcut
      is worth nothing, and the ladder runs sat at exactly 50.0%.
    - The shortcut may be a stepping stone: a head that already attends to the listed values only has
      to learn which one to pick.
- **Next.** The check that follows is pre-registered in `plan_formula_evaluation.md`. It runs the
  trainer on run E's exact pool, adds balanced queries to run E, and repeats run E with a second seed.

## Trainer, balanced queries or seed: results

This is the check pre-registered in `plan_formula_evaluation.md`. Every run has run E's conditions: 21
facts, a fixed pool of 50,000, the cosine schedule and 31,000 steps.

| run | training queries | seed | result |
|---|---|---|---|
| E (factor check) | uniform | 0 | **jump at step 15,500, then 1.000** |
| E1 | uniform | 1 | partial: 0.59-0.60 to step 9,300, then 0.64-0.69 from step 12,400 (loss 0.53-0.56); no full jump |
| T: `training/train.py` on run E's exact pool | uniform | 0 (trainer) | partial: test_iid 67.4%. Validation loss 0.664 for 10 epochs, then down to 0.52 by epoch 17, flat after |
| F | **balanced** | 0 | **exactly chance throughout**: 0.47-0.54, loss 0.691-0.696 |

**Against the pre-registration:**

- **Predictions:**
  - "T is learned": false. It reached 67.4%, against a criterion of 95%.
  - "F is not learned": true.
  - "E1 is learned": false.
- **Decision rule:**
  - **E1 failed.** So the timing of the jump at 21 facts varies widely between seeds, and runs that
    decide anything need 3 seeds.
  - **T failed too.** The T clause names the trainer, but T's curve matches E1's: a partial solution
    with loss around 0.52-0.55. A second seed of the sanity script does the same without the trainer.
    So T is not evidence against the trainer. The 3-seed ladder runs will show whether the trainer
    matters at all.
  - **The balanced-query clause** was written for "T learns and F fails". T did not learn. F's result
    stands on its own all the same.

**Reading:**

- **Balanced training queries are the ladder's obstacle.** F differs from run E only in how training
  queries are drawn. It never leaves exact chance in 31,000 steps, the same flat 0.693 as every ladder
  run. All three runs with uniform queries left chance: E fully, E1 and T partly.
  - With uniform queries, answering with the table's majority value beats chance, and every run that
    learned sat on that shortcut first. Balanced queries make the shortcut worth nothing, and the
    gradient never finds the lookup.
- **At 21 facts the lookup forms in only some seeds within 31,000 steps.** Of the three uniform-query
  runs, one got the full jump. The other two found a partial solution (about 67%) and stayed there.
- **Fix.** `data/eval_generator.py --train-query-sampling uniform` draws training queries uniformly over
  all atoms. Val and test items are unchanged: balanced, with twins. The next check is pre-registered in
  `plan_formula_evaluation.md`.

## Uniform training queries, three seeds: results

This is the check pre-registered in `plan_formula_evaluation.md`, run on the AMD machine: a GMKtec EVO-X2
with Radeon integrated graphics, under ROCm. Its smoke test reproduces the laptop: `pairs` jumps at step
5,000 on both machines.

**Part 1: `sanity_recall.py`.** `table_keylast`, 21 facts, a fixed pool of 50,000, the cosine schedule,
31,000 steps, uniform queries:

| queries per sequence | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| 8 | **jump at 15,500** (run E) | partial: about 0.65-0.69, loss about 0.53 (E1) | partial: 0.65-0.70, loss 0.53-0.57 |
| 16 | **jump by 12,400** | partial: 0.65-0.71, loss 0.54-0.56 | **jump by 12,400** |

**Part 2: the ladder's `single` rung.** Key-last, 8 uniform training queries, 40 epochs. test_iid
accuracy / pair accuracy (%):

| seed | test_iid | by objects (12 / 21 / 32) | validation loss |
|---|---|---|---|
| 1 | 80.0 / 60.1 | 74.2 / 87.7 / 78.7 | 0.678, falling from epoch 11 to a partial 0.55 by epoch 26; a second fall starts at epoch 38 (0.534, 0.474, 0.377) as training ends |
| 2 | 66.6 / 33.3 | 74.8 / 64.3 / 61.1 | 0.679, down to a partial 0.53 by epoch 21, flat after |
| 3 | **100.0 / 100.0** | 100 / 100 / 100 | 0.681, then at epoch 20 from 0.565 to 0.125, and 0.0000 from epoch 33 |

**Against the pre-registration:**

- **Predictions:**
  - Part 1, 8 queries, seed 2, was left open: partial.
  - Part 1, "16 queries learned in all three seeds": false (2 of 3).
  - Part 2, "every seed leaves exact chance": true. All three fell below 0.69, to 0.53 or lower.
  - Part 2, reaching 95% pairs in at least one seed was left open: seed 3 reached 100%.
  - Part 2, "12-object worlds learned before 32-object ones": holds for seed 2. Seed 1 is best at 21.
- **Decision rule.** Part 2 passes in 1 seed, and Part 1 did not find 16 queries reliable. So the next
  step is the size curriculum: small worlds first, then a warm start on the full rung.

**Reading:**

- **Uniform training queries turn a lookup that was never learned into one that sometimes is.** For the
  first time a ladder run is perfect at every world size, 12 to 32 objects.
- **Runs that do not jump all stall at the same partial solution:** loss about 0.53, accuracy about 67%.
  That holds in the sanity script and in the trainer, at 8 and at 16 queries. It looks like a separate
  attractor, not a slow approach to the full solution. Seed 1 escaped it late.
- **More queries per sequence help.** At 16, the jump came earlier (by step 12,400 against 15,500) and in
  more seeds (2 of 3 against 1 of 3). They do not make it reliable.

## Size curriculum, three seeds: results

This is the check pre-registered in `plan_formula_evaluation.md`, run on the AMD machine. Stage 1 is 20
epochs on the `single` rung with 4-8 objects. Stage 2 is 20 epochs on the full rung (12-32 objects),
warm-started from stage 1. Columns: test_iid accuracy / pair accuracy (%).

| seed | stage 1 (4 / 6 / 8 objects) | stage 1 validation loss | stage 2 (12 / 21 / 32 objects) | stage 2 validation loss |
|---|---|---|---|---|
| 1 | 100.0 / 100.0 (100 / 100 / 100) | 0.597, 0.594, 0.598, 0.576, then 0.099 in epoch 5 | **100.0 / 100.0** (100 / 100 / 100) | 0.343, 0.127, 0.054, then 0.000 |
| 2 | 100.0 / 100.0 (100 / 100 / 100) | to 0.531 in epoch 5, then 0.030 in epoch 6 | **100.0 / 100.0** (100 / 100 / 100) | 0.359, 0.193, 0.066, 0.034, then 0.000 |
| 3 | 100.0 / 100.0 (100 / 100 / 100) | to 0.505 in epoch 5, then 0.002 in epoch 6 | **100.0 / 100.0** (100 / 100 / 100) | 0.349, 0.252, 0.107, 0.016, then 0.000 |

**Against the pre-registration:**

- **Predictions:**
  - "Stage 1 is learned in all 3 seeds": true. Each jump came in epoch 5 or 6, about 4,000-4,700
    steps in.
  - "Stage 2 passes in all 3 seeds, with validation loss below 0.1 within 5 epochs": true. Validation
    loss was below 0.1 by epoch 3 or 4.
- **Decision rule.** The curriculum is the recipe. Next are the `unary` and `full` rungs, small worlds
  first, 3 seeds each.

**Reading:**

- **One-token lookups are solved.** Every seed is at 100% on every world size, with the same 40-epoch
  budget under which training from scratch succeeded in 1 seed of 3. The partial solution never
  appears: on small tables the full lookup forms before anything else can settle in.
- **What is learned on small tables carries over to large ones almost at once.** Stage 2 starts at a
  validation loss of about 0.35, far below the 0.69 of chance, and reaches 0 within 2-4 epochs. The
  head matches keys whatever the table's size, as predicted.
- **The full recipe so far:**
  - each value right after its key, and the answer predicted at the query key (key-last);
  - training queries drawn uniformly over the facts;
  - small worlds first.
