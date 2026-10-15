# Withdrawal routes: every way off the residual, on one task

Companion to `magnitude_precision.md` (the study that found the fade) and `plan_distinctive_result.md`
(the plan this follows). Same data as the precision study (values 0-499, 350-449 held out, 500-624 never
seen, 40% close pairs), copied to `runs/magnitude_precision_routes/data`; same tiny model; three seeds
per arm unless stated. Results are added milestone by milestone.

## M0: measuring readability instead of asserting it

Two new measurements on saved models (`evaluate_magnitude.py --residual-scale`, `scripts/residual_scale_sweep.py`,
`scripts/psychometric_split.py`, `scripts/psychometric_curves.py`).

### Readability curves

A saved grid_shared model is re-evaluated with its per-cell residual multiplied by a scale from 0 to 1 (the
override is applied after the orthogonal projection on every forward pass; the test
`test_evaluator_residual_scale_override_matches_zeroed_residual` checks that scale 0 equals a zeroed
residual row for row). A comparator that reads the ramp is flat; one that still reads the residual falls
off below some scale. Sanity check from the plan (accuracy, %):

| model | saved scale | split | 0 | 1e-6 | 1e-5 | 1e-4 | 1e-3 | 1e-2 | 0.1 | 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| deep anneal, seed 1 | 1e-6 | test | 99.0 | 99.0 | 99.1 | 98.8 | 96.4 | 72.4 | 47.7 | 52.6 |
| deep anneal, seed 1 | 1e-6 | held-out | 98.2 | 98.2 | 98.3 | 97.7 | 91.3 | 65.5 | 50.6 | 49.4 |
| deep anneal, seed 1 | 1e-6 | far | 72.3 | 72.3 | 73.7 | 76.3 | 70.0 | 56.2 | 48.6 | 51.3 |
| anneal and hold, seed 1 | 0.01 | test | 51.7 | 51.7 | 51.7 | 51.7 | 52.0 | 99.3 | 88.4 | 51.7 |
| anneal and hold, seed 1 | 0.01 | held-out | 50.4 | 50.4 | 50.4 | 50.8 | 53.6 | 97.5 | 78.5 | 48.9 |
| anneal and hold, seed 1 | 0.01 | far | 50.2 | 50.2 | 50.2 | 50.2 | 51.7 | 89.9 | 72.4 | 49.2 |

The numbers the plan asked for reproduce (99.0 / 98.2 / 72.3 at 1e-6 and at 0 for the deep model; chance at
0 for the hold model). Two things the table adds: the hold model reads its residual only in a window around
the scale it was trained at (chance at 1e-3, 97.5% at 1e-2, 78.5% at 0.1, chance at 1), and the deep model
is not merely indifferent to the residual but hurt by it from 1e-3 up: a comparator tuned to the ramp has no
use for per-cell texture, and at full scale that texture dominates the normalised embedding.

%%SWEEP_ALL%%

### Psychometric decision curves

Every k from 0 to 624 is compared with a fixed c (375 and 425 inside the held-out band, 520 and 580 past
the trained range) under both operators; the model's P(TRUE) as a function of k is its decision curve.
"Transition width" is the number of k with 0.25 < P(TRUE) < 0.75; "errors" are split by where k lies
(trained values / held-out band / past the range). Means over three seeds:

| model | op | c | accuracy (%) | transition width | errors train / held-out / far |
|---|---|---|---|---|---|
| deep anneal (residual-free) | LESS | 375 | 99.8 | 6 | 0 / 1 / 0 |
| deep anneal (residual-free) | LESS | 425 | 99.6 | 12 | 0 / 2 / 0 |
| deep anneal (residual-free) | LESS | 520 | 97.2 | 14 | 1 / 0 / 16 |
| deep anneal (residual-free) | LESS | 580 | 91.9 | 30 | 2 / 0 / 49 |
| deep anneal (residual-free) | GREATER | 375 | 95.8 | 13 | 0 / 1 / 25 |
| deep anneal (residual-free) | GREATER | 580 | 91.1 | 72 | 2 / 0 / 54 |
| anneal and hold (residual 0.01) | LESS | 375 | 99.9 | 8 | 0 / 0 / 0 |
| anneal and hold (residual 0.01) | LESS | 425 | 99.8 | 14 | 0 / 1 / 0 |
| anneal and hold (residual 0.01) | LESS | 520 | 99.4 | 122 | 0 / 0 / 4 |
| anneal and hold (residual 0.01) | LESS | 580 | 97.0 | 119 | 0 / 0 / 19 |
| anneal and hold (residual 0.01) | GREATER | 520 | 99.8 | 128 | 0 / 0 / 1 |
| dropout 0.5, 80 epochs | LESS | 375 | 92.2 | 77 | 4 / 44 / 0 |
| dropout 0.5, 80 epochs | LESS | 425 | 90.6 | 73 | 4 / 55 / 0 |
| dropout 0.5, 80 epochs | LESS | 520 | 99.6 | 61 | 0 / 0 / 3 |
| dropout 0.5, 80 epochs | LESS | 580 | 99.6 | 47 | 0 / 0 / 3 |
| smooth sinusoid features | LESS | 375 | 68.9 | 30 | 41 / 68 / 86 |
| smooth sinusoid features | LESS | 580 | 33.8 | 40 | 263 / 100 / 51 |

(The full table, every seed, value and operator, is in `runs/magnitude_precision_routes/psycho/curves.json`.)

Reading:

- Inside the never-seen band the residual-free comparator's threshold is at unit resolution: the curve
  crosses 0.5 within two units of c and the transition is 6-12 values wide, at a c no training pair ever
  contained. The hold model matches it there.
- Past the trained range the two part company. The deep model keeps a 14-30 wide transition at c = 520 and
  580; the hold model's smears to 119-128, so its high accuracy there is only the fact that most k are far
  from c. Resolution past the range is a property of the ramp-only comparator that the residual destroys.
- The deep model's far-band accuracy is held down by a bias, not by resolution: for k past the range it
  answers FALSE to both "k LESS_THAN c" and "k GREATER_THAN c", so every GREATER_THAN error at c = 375 and
  425 is a far-band k (25-26 of 125), while LESS_THAN has none. An out-of-range operand reads as "neither".
- The dropout arm is the mirror image: no resolution inside the hole (73-86 wide, 44-55 held-out errors)
  and good resolution past the range, which is the interpolation / extrapolation split in its accuracy
  tables made visible. The sinusoid features fail everywhere, with transition widths that mean nothing
  because the curves are not monotone.
