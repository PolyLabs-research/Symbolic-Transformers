# Numeral codes head to head: our ramp against an xVal-style code, on two tasks

**Status: runs in progress (interim tables below).** Tables are regenerated with `python scripts/numeral_codes_table.py`.

The question this answers: after `withdrawal_routes.md` reduced our numeral code to "a generated
two-direction thermometer code, LayerNormed, with learned operator rows", is it any better than the
established simple alternative, an xVal-style code (Golkar et al. 2023: one learned direction scaled by
the number's standardised value), and does either support arithmetic on values never seen in training?

## Setup

Same tiny post-LayerNorm transformer for every arm (d = 128, 2 layers, embeddings scaled by √d, sinusoidal
positions, tied output), 120 epochs, learning rate 3e-4, three seeds. Values 0-499 are trained except the
held-out band 350-449; 500-624 are never seen. Operators, PLUS and the answers are learned rows in every arm.

| arm | numeral k is embedded as |
|---|---|
| learned table (`learned_untied`) | its own learned row |
| smooth sinusoid features (`sinusoid_smooth`) | fixed sin/cos of k at periods 32-1250, through a learned linear layer and LayerNorm |
| xVal-style (`xval`) | s(k) · h, with h one learned direction and s(k) = (k − 249.5) / 144.3 (standardised over 0-499); no LayerNorm on numeral rows, so the scaling reaches the first attention layer |
| ramp, numerals only (`grid_ramp_numonly`, ours) | the thermometer panel of k through two generated directions (k·a + k²/1250·c + bias), then LayerNorm |

Tasks (same value pools; `data/magnitude_generator.py --task compare|sum`):

- **Comparison:** `NUM_a OP NUM_b → TRUE/FALSE`, OP in LESS_THAN / GREATER_THAN. Data: `runs/magnitude_precision_routes/data`.
- **Sum comparison:** `NUM_a PLUS NUM_b OP NUM_c PLUS NUM_d → TRUE iff (a + b) OP (c + d)`. In the
  held-out and far splits half the samples have all four operands unseen. Data: `runs/numeral_codes_sum/data`.

## How to read the tables (from an independent review of the code and data)

- **Rank arms on the all-unseen columns.** About half of each unseen split can be answered without knowing
  the unseen values: treating every unseen operand as "big" gets 66% of the held-out split and 68% of the
  far split on the sum task. On the all-unseen subsets any constant guess is at the base rate (51-52%).
- **Compare tasks only within gap bins.** The gap mixes differ: on the comparison task 73% of held-out
  all-unseen samples have gap 1-10 and 18% gap above 20; on the sum task 41% and 48%. Far all-unseen sums
  (1000-1248) lie wholly outside the training sum range (7-993); held-out ones (700-898) lie inside it.
- **Read the escape-epoch column before the accuracies.** A seed that never leaves chance is a training
  outcome at this schedule, not evidence about what the code can represent; the controls below separate
  the two.
- The embedding probes in `results.md` ("is magnitude in the embedding?") are meaningless for the xVal
  code: its numeral block is rank one by construction and the probes are perfect at initialisation.
- `runs/*/results.md` and `config.json` are rewritten by every driver call into the directory; the per-run
  `eval.json` files and this table are the record.

## Results

Accuracy (%), mean ± population std over seeds. Control rows belong to the next section.

### comparison: NUM_a OP NUM_b

| numeral code | seeds | seen | seen gap 1 / 2-5 | held-out, all operands unseen (gap 1 / 2-5 / 6-20) | far, all unseen | held-out band | far band | TRUE/FALSE only: held-out / far | escape epoch per seed |
|---|---|---|---|---|---|---|---|---|---|
| learned table | 0 | (not run yet) | | | | | | | |
| smooth sinusoid features | 3 | 100.0 ± 0.0 | 100.0 / 100.0 | 46.2 ± 9.5 (54.3 / 54.4 / 47.7) | 64.4 ± 9.5 | 70.7 ± 2.6 | 59.5 ± 3.6 | 70.7 / 59.5 | 2, 2, 3 |
| xVal-style (value x one direction) | 3 | 99.4 ± 0.3 | 93.0 / 99.2 | 96.3 ± 2.8 (76.2 / 96.5 / 100.0) | 94.5 ± 2.0 | 98.2 ± 1.4 | 97.2 ± 1.1 | 98.2 / 97.2 | 2, 2, 2 |
| ramp, numerals only (ours) | 3 | 99.3 ± 0.2 | 89.9 / 97.7 | 92.0 ± 0.9 (57.0 / 85.8 / 99.3) | 80.0 ± 1.6 | 96.0 ± 0.5 | 89.8 ± 0.8 | 96.0 / 89.8 | 6, 9, 7 |

### sum comparison: NUM_a + NUM_b OP NUM_c + NUM_d

| numeral code | seeds | seen | seen gap 1 / 2-5 | held-out, all operands unseen (gap 1 / 2-5 / 6-20) | far, all unseen | held-out band | far band | TRUE/FALSE only: held-out / far | escape epoch per seed |
|---|---|---|---|---|---|---|---|---|---|
| learned table | 0 | (not run yet) | | | | | | | |
| smooth sinusoid features | 0 | (not run yet) | | | | | | | |
| xVal-style (value x one direction) | 2 | 94.0 ± 0.4 | 63.0 / 76.1 | 76.3 ± 0.6 (46.8 / 57.9 / 67.4) | 56.0 ± 7.2 | 84.0 ± 0.7 | 70.3 ± 4.4 | 84.0 / 70.3 | 2, 2 |
| ramp, numerals only (ours) | 2 | 50.3 ± 0.0 | 55.9 / 52.9 | 50.9 ± 0.0 (51.1 / 48.1 / 50.2) | 51.9 ± 0.0 | 49.8 ± 0.0 | 51.0 ± 0.0 | 49.8 / 51.0 | never, never |
| control: xVal-style, lr 1e-3 | 0 | (not run yet) | | | | | | | |
| control: ramp, lr 1e-3 | 0 | (not run yet) | | | | | | | |
| control: ramp, warm start from the comparison task | 0 | (not run yet) | | | | | | | |

Interim reading (comparison task complete for all codes but the learned table; sum task one xVal seed and two ramp seeds):

- **On the comparison task the xVal-style code beats ours on every unseen measure.** All-unseen held-out
  pairs 96.3 ± 2.8% against 92.0 ± 0.9% (gap 1: 76% against 57%), all-unseen far pairs 94.5 ± 2.0%
  against 80.0 ± 1.6%, with seen pairs equal (99.4 / 99.3) and every seed leaving chance by epoch 2
  against 6-9. Its far band is the best result of any model in this project, the 80-epoch residual-dropout
  arm of the precision study included (93% far all-unseen, but 52% held-out all-unseen).
- **On the sum task, so far only the xVal-style code learns.** Seed 1: 93.6% on seen sums, 75.7% on
  held-out samples with all four operands unseen, 63.2% on far ones. Our ramp's first two seeds never left
  chance in 120 epochs. Whether that is the code or the optimisation is what the controls decide.

## Controls for a stall on the sum task

Our ramp sat at chance on the sum task for at least the first 40 epochs of its first seed, while the
xVal-style code left chance by epoch 3. Two controls, three seeds each, decide whether that is optimisation
or representation:

- **Warm start** (`grid_ramp_numonly_warm`): the ramp model trained on the comparison task, fine-tuned on
  sums with a fresh optimiser and schedule. Its comparator already reads magnitude from the code, so if the
  code can represent sums this arm should learn them.
- **Hotter schedule** (`grid_ramp_numonly_lr1e3`, and `xval_lr1e3` for parity): learning rate 1e-3.

Not run yet.
