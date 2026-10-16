# Numeral codes head to head: our ramp against an xVal-style code, on two tasks

**Status: complete** (24 main runs over both tasks, of which the comparison-task ramp and sinusoid rows reuse runs from `withdrawal_routes.md` and `magnitude_precision.md`, plus 12 control runs). Tables are regenerated with `python scripts/numeral_codes_table.py`; per-seed numbers are in `numeral_codes.json`.

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
| learned table | 3 | 97.1 ± 0.3 | 71.3 / 96.7 | 49.4 ± 1.3 (47.5 / 47.1 / 48.6) | 52.2 ± 1.6 | 55.2 ± 0.2 | 50.2 ± 1.0 | 55.2 / 50.2 | 7, 5, 4 |
| smooth sinusoid features | 3 | 100.0 ± 0.0 | 100.0 / 100.0 | 46.2 ± 9.5 (54.3 / 54.4 / 47.7) | 64.4 ± 9.5 | 70.7 ± 2.6 | 59.5 ± 3.6 | 70.7 / 59.5 | 2, 2, 3 |
| xVal-style (value x one direction) | 3 | 99.4 ± 0.3 | 93.0 / 99.2 | 96.3 ± 2.8 (76.2 / 96.5 / 100.0) | 94.5 ± 2.0 | 98.2 ± 1.4 | 97.2 ± 1.1 | 98.2 / 97.2 | 2, 2, 2 |
| ramp, numerals only (ours) | 3 | 99.3 ± 0.2 | 89.9 / 97.7 | 92.0 ± 0.9 (57.0 / 85.8 / 99.3) | 80.0 ± 1.6 | 96.0 ± 0.5 | 89.8 ± 0.8 | 96.0 / 89.8 | 6, 9, 7 |

### sum comparison: NUM_a + NUM_b OP NUM_c + NUM_d

| numeral code | seeds | seen | seen gap 1 / 2-5 | held-out, all operands unseen (gap 1 / 2-5 / 6-20) | far, all unseen | held-out band | far band | TRUE/FALSE only: held-out / far | escape epoch per seed |
|---|---|---|---|---|---|---|---|---|---|
| learned table | 3 | 82.0 ± 0.4 | 56.7 / 48.8 | 49.8 ± 0.8 (52.1 / 48.5 / 48.8) | 48.7 ± 1.6 | 57.9 ± 0.9 | 53.8 ± 1.3 | 57.9 / 53.8 | 10, 6, 6 |
| smooth sinusoid features | 3 | 93.0 ± 0.5 | 59.8 / 71.5 | 73.1 ± 2.9 (52.5 / 56.9 / 61.5) | 18.2 ± 2.9 | 79.2 ± 2.0 | 42.1 ± 1.9 | 79.2 / 42.1 | 4, 5, 3 |
| xVal-style (value x one direction) | 3 | 93.6 ± 0.7 | 61.4 / 74.3 | 76.9 ± 1.0 (47.5 / 58.9 / 68.1) | 54.5 ± 6.3 | 83.6 ± 0.9 | 69.6 ± 3.7 | 83.6 / 69.6 | 2, 2, 3 |
| ramp, numerals only (ours) | 3 | 50.3 ± 0.0 | 55.9 / 52.9 | 50.9 ± 0.0 (51.1 / 48.1 / 50.2) | 51.9 ± 0.0 | 49.8 ± 0.0 | 51.0 ± 0.0 | 49.8 / 51.0 | never, never, never |
| control: xVal-style, lr 1e-3 | 3 | 92.8 ± 0.8 | 61.9 / 71.4 | 79.7 ± 2.4 (53.9 / 57.9 / 70.1) | 53.9 ± 8.3 | 84.2 ± 1.1 | 68.9 ± 4.8 | 84.2 / 68.9 | 2, 2, 2 |
| control: ramp, lr 1e-3 | 3 | 50.3 ± 0.0 | 55.9 / 52.9 | 50.9 ± 0.0 (51.1 / 48.1 / 50.2) | 51.9 ± 0.0 | 49.8 ± 0.0 | 51.0 ± 0.0 | 49.8 / 51.0 | never, never, never |
| control: ramp, warm start from the comparison task | 3 | 93.8 ± 1.0 | 62.5 / 76.0 | 85.1 ± 4.8 (48.6 / 64.2 / 83.1) | 51.0 ± 6.3 | 87.9 ± 2.4 | 67.4 ± 3.6 | 87.9 / 67.4 | 1, 1, 1 |
| control: xVal-style, warm start from the comparison task | 3 | 94.3 ± 0.1 | 61.7 / 77.7 | 79.1 ± 2.8 (46.5 / 53.7 / 70.0) | 71.9 ± 1.5 | 84.1 ± 1.6 | 79.6 ± 0.9 | 84.1 / 79.6 | 1, 1, 1 |

Reading of the main runs (three seeds each; per-seed values in `numeral_codes.json`):

- **On the comparison task the xVal-style code beats ours on every unseen measure.** All-unseen held-out
  pairs 96.3 ± 2.8% against 92.0 ± 0.9% (seeds 93.2 / 95.8 / 100.0 against 90.8 / 92.0 / 93.1; gap 1:
  76% against 57%, both with wide seed spread), all-unseen far pairs 94.5 ± 2.0% against 80.0 ± 1.6% (no
  seed overlap), with seen pairs equal (99.4 / 99.3) and every seed leaving chance by epoch 2 against 6-9.
  Its far band (97.2 ± 1.1%) matches the best earlier model in the project, the 80-epoch residual-dropout
  arm of the precision study (96.3 ± 1.9%; all-unseen far pairs 94.5% against 92.8%), and unlike that arm
  it keeps the held-out band (all-unseen 96.3% against 52%). The learned table fits seen pairs (97.1%) and
  is at chance on every unseen subset; the smooth sinusoid features fit seen pairs perfectly and are at or
  below chance on held-out all-unseen pairs (46.2 ± 9.5%).
- **On the sum task, from scratch, only the linear codes interpolate and nothing extrapolates reliably.**
  The xVal-style code reaches 93.6% on seen sums and 76.9 ± 1.0% on held-out samples with all four
  operands unseen, but 54.5 ± 6.3% on far ones (seeds 63.2 / 48.8 / 51.6), whose sums (1000-1248) all lie
  above the training range (7-993). The sinusoid features interpolate nearly as well (73.1 ± 2.9%) and are
  inverted past the range (18.2 ± 2.9%). The learned table fits seen sums to 82% and is at chance on unseen
  ones. Our ramp never left chance in 120 epochs in any of three seeds: every run outputs the same
  P(TRUE) ≈ 0.47 for every input (always FALSE), which is why its rows are identical across seeds.
- **Ranking from scratch, all-unseen columns, both tasks:** xVal-style first everywhere; our ramp second on
  the comparison task and last on the sum task; sinusoid features interpolate sums but invert past the
  range; the learned table never generalises. The controls below show the ramp's sum-task result is an
  optimisation failure, not a representational one, and narrow but do not reverse the ranking.

## Controls for a stall on the sum task

Our ramp sat at chance on the sum task for at least the first 40 epochs of its first seed, while the
xVal-style code left chance by epoch 3. Two controls decide whether that is optimisation or representation,
and each has an xVal-style twin for parity (three seeds per arm):

- **Warm start** (`grid_ramp_numonly_warm`): the ramp model trained on the comparison task, fine-tuned on
  sums with a fresh optimiser and schedule. Its comparator already reads magnitude from the code, so if the
  code can represent sums this arm should learn them.
- **Hotter schedule** (`grid_ramp_numonly_lr1e3`, and `xval_lr1e3` for parity): learning rate 1e-3.
- **Matched warm start** (`xval_warm`): the xVal-style model trained on the comparison task, fine-tuned on sums
  the same way. Without it a win for the warm-started ramp could be the curriculum rather than the code.

All 12 control runs complete (three seeds per arm).

- **The ramp can represent sums; what fails from scratch is optimisation.** Started from its
  comparison-task checkpoint, our ramp leaves chance in the first epoch of all three seeds and ends at
  93.8 ± 1.0% on seen sums and 85.1 ± 4.8% on held-out samples with all four operands unseen (seeds
  88.2 / 88.7 / 78.3). A hotter schedule does not rescue it: at learning rate 1e-3 all three seeds again
  sit at the constant always-FALSE output for 120 epochs, while the xVal-style code at 1e-3 learns as
  before (79.7 ± 2.4% held-out all-unseen, 53.9 ± 8.3% far).
- **With matched warm starts the two codes split the unseen columns.** The parity control, xVal
  warm-started from its own comparison checkpoint (`xval_warm`), ends at 94.3 ± 0.1% seen, 79.1 ± 2.8%
  held-out all-unseen (seeds 76.6 / 77.6 / 83.0) and 71.9 ± 1.5% far all-unseen (70.2-73.9).
  - Held-out (interpolation): the ramp leads by about 6 points, 85.1 against 79.1. The lead is at moderate
    gaps (2-5: 64.2 against 53.7; 6-20: 83.1 against 70.0); at gap 1 both are at chance (48.6 / 46.5). Two
    ramp seeds are above every xVal seed, but the third (78.3) is below xVal's best (83.0), so with three
    seeds this is a lead, not a settled one.
  - Far (sums past the trained range): xVal leads by about 21 points, 71.9 against 51.0 ± 6.3 (ramp seeds
    42.3 / 56.9 / 53.9, near chance; no seed overlap). The warm start lifts xVal's far result from 54.5 to
    71.9, so a linear code with a comparator trained on magnitude first does extrapolate sums part of the
    way; the ramp does not.
- **Verdict.** The xVal-style code is the better default: it trains from scratch on both tasks, leads on
  every unseen measure of the comparison task, and is the only code that reaches past the trained range on
  sums. Our ramp's one advantage is a modest, seed-dependent edge on interpolating sums, and only after a
  curriculum that xVal does not need. Neither code resolves neighbouring unseen sums (gap 1).
