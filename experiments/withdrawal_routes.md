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

Accuracy (%) on held-out band, both operands unseen, every existing model re-evaluated at each residual scale (mean ± std over seeds;
per-seed numbers in `runs/magnitude_precision_routes/sweeps/existing_models.json`):

| trained as | saved scale | 0 | 1e-06 | 1e-05 | 0.0001 | 0.001 | 0.01 | 0.1 | 1 |
|---|---|---|---|---|---|---|---|---|---|
| residual at full scale, 40 epochs | 1 | 35 ± 21 | 35 ± 21 | 35 ± 21 | 36 ± 20 | 39 ± 15 | 50 ± 1 | 52 ± 1 | 76 ± 5 |
| residual dropout 0.5, 40 epochs | 1 | 50 ± 1 | 50 ± 1 | 50 ± 1 | 50 ± 1 | 50 ± 1 | 50 ± 1 | 51 ± 0 | 84 ± 4 |
| residual dropout 0.5, 80 epochs | 1 | 51 ± 0 | 51 ± 0 | 51 ± 0 | 51 ± 0 | 51 ± 0 | 49 ± 1 | 50 ± 1 | 52 ± 16 |
| linear anneal to zero (collapsed) | 0 | 49 ± 0 | 49 ± 0 | 49 ± 0 | 49 ± 0 | 49 ± 0 | 50 ± 1 | 51 ± 0 | 49 ± 0 |
| geometric anneal to zero, 80 epochs | 0 | 66 ± 23 | 66 ± 23 | 66 ± 23 | 66 ± 23 | 63 ± 23 | 57 ± 9 | 51 ± 0 | 49 ± 0 |
| geometric anneal to zero, 120 epochs | 0 | 66 ± 23 | 66 ± 23 | 66 ± 23 | 66 ± 24 | 66 ± 22 | 59 ± 12 | 50 ± 1 | 49 ± 0 |
| geometric anneal + LR re-warm-up (collapsed) | 0 | 50 ± 1 | 50 ± 1 | 50 ± 1 | 50 ± 1 | 51 ± 0 | 51 ± 0 | 49 ± 0 | 49 ± 0 |
| anneal and hold at 0.01 | 0.01 | 50 ± 1 | 50 ± 1 | 50 ± 1 | 51 ± 2 | 58 ± 4 | 95 ± 4 | 56 ± 5 | 49 ± 0 |
| deep anneal to 1e-6 (residual-free) | 1e-06 | 95 ± 2 | 95 ± 2 | 96 ± 2 | 95 ± 3 | 84 ± 9 | 63 ± 8 | 51 ± 0 | 49 ± 0 |
| ramp alone from scratch (no residual) | none | 50 ± 1 | 50 ± 1 | 50 ± 1 | 50 ± 1 | 50 ± 1 | 50 ± 1 | 50 ± 1 | 50 ± 1 |
| smooth sinusoid features (no residual) | none | 46 ± 9 | 46 ± 9 | 46 ± 9 | 46 ± 9 | 46 ± 9 | 46 ± 9 | 46 ± 9 | 46 ± 9 |

Accuracy (%) on seen pairs, every existing model re-evaluated at each residual scale (mean ± std over seeds;
per-seed numbers in `runs/magnitude_precision_routes/sweeps/existing_models.json`):

| trained as | saved scale | 0 | 1e-06 | 1e-05 | 0.0001 | 0.001 | 0.01 | 0.1 | 1 |
|---|---|---|---|---|---|---|---|---|---|
| residual at full scale, 40 epochs | 1 | 36 ± 7 | 36 ± 7 | 36 ± 7 | 36 ± 7 | 37 ± 6 | 41 ± 5 | 48 ± 0 | 98 ± 1 |
| residual dropout 0.5, 40 epochs | 1 | 51 ± 2 | 51 ± 2 | 51 ± 2 | 51 ± 2 | 51 ± 3 | 51 ± 2 | 47 ± 0 | 99 ± 0 |
| residual dropout 0.5, 80 epochs | 1 | 47 ± 0 | 47 ± 0 | 47 ± 0 | 47 ± 0 | 47 ± 0 | 50 ± 2 | 50 ± 3 | 99 ± 0 |
| linear anneal to zero (collapsed) | 0 | 53 ± 0 | 53 ± 0 | 53 ± 0 | 53 ± 0 | 52 ± 0 | 50 ± 4 | 48 ± 0 | 54 ± 1 |
| geometric anneal to zero, 80 epochs | 0 | 69 ± 22 | 69 ± 22 | 69 ± 22 | 68 ± 22 | 65 ± 24 | 56 ± 13 | 48 ± 1 | 53 ± 0 |
| geometric anneal to zero, 120 epochs | 0 | 68 ± 23 | 68 ± 22 | 68 ± 22 | 68 ± 22 | 65 ± 24 | 55 ± 8 | 51 ± 2 | 54 ± 1 |
| geometric anneal + LR re-warm-up (collapsed) | 0 | 51 ± 3 | 51 ± 3 | 51 ± 3 | 51 ± 3 | 47 ± 0 | 47 ± 0 | 53 ± 0 | 53 ± 0 |
| anneal and hold at 0.01 | 0.01 | 52 ± 1 | 52 ± 1 | 52 ± 1 | 53 ± 2 | 70 ± 17 | 99 ± 0 | 80 ± 8 | 54 ± 2 |
| deep anneal to 1e-6 (residual-free) | 1e-06 | 100 ± 0 | 100 ± 0 | 100 ± 0 | 99 ± 0 | 98 ± 1 | 76 ± 5 | 48 ± 0 | 51 ± 2 |
| ramp alone from scratch (no residual) | none | 52 ± 0 | 52 ± 0 | 52 ± 0 | 52 ± 0 | 52 ± 0 | 52 ± 0 | 52 ± 0 | 52 ± 0 |
| smooth sinusoid features (no residual) | none | 100 ± 0 | 100 ± 0 | 100 ± 0 | 100 ± 0 | 100 ± 0 | 100 ± 0 | 100 ± 0 | 100 ± 0 |
Reading the all-rows curves (every grid-backed row scaled together):

- Trained with the residual at full scale (plain, or with dropout), the comparator reads it only at full
  scale: at 0.1 every such model is at chance and at 0 the 40-epoch plain models are inverted on seen pairs
  (27-41%) and one is at 6% on both-unseen held-out pairs.
- Annealed and held at 0.01, the comparator reads a window around the scale it was trained at: chance at
  1e-3, 90-100% at 1e-2, 51-63% at 0.1, chance at 1.
- Faded below readability (deep arm) or recovered from the step (the two lucky geometric seeds), the
  comparator is flat from 0 to 1e-4, degrades from 1e-3 and is at chance by 0.1.
- Collapsed seeds and the residual-free codes are flat at chance (or, for the sinusoid features, flat at
  their memorised seen-pair score) across the whole axis.

### The confound: the residual also carries the operator distinction

The adversarial review of these scripts found what the all-rows curves actually measure. The operator
tokens LESS_THAN and GREATER_THAN and the answer tokens TRUE and FALSE are grid panels that go through the
same shared encoder and the same per-cell residual as the numerals, and under the ramp basis (two shared
directions) the generated parts of the two operator panels are nearly identical: after LayerNorm their
difference has norm 0.005 on a row of norm 1.5-2.1 (cosine 0.999996). The operator distinction therefore
lives in the residual. The per-row override (`--residual-scale-rows numerals|glyphs`) separates the two
effects. Accuracy (%) with only the numeral rows re-scaled, the four glyph rows kept at the saved scale:

Seen pairs (test):

| model | 0 | 1e-06 | 1e-05 | 0.0001 | 0.001 | 0.01 | 0.1 | 1 |
|---|---|---|---|---|---|---|---|---|
| anneal and hold, seed 1 (saved 0.01) | 98.8 | 98.8 | 98.8 | 98.8 | 98.8 | 99.3 | 90.4 | 51.7 |
| deep anneal, seed 1 (saved 1e-6) | 99.0 | 99.0 | 99.0 | 99.0 | 99.1 | 97.5 | 66.4 | 51.2 |
| residual dropout, 80 epochs, seed 1 (saved 1) | 2.0 | 2.0 | 2.0 | 2.0 | 2.0 | 2.0 | 2.1 | 99.7 |
| residual at full scale, 40 epochs, seed 1 (saved 1) | 50.4 | 50.4 | 50.4 | 50.4 | 50.4 | 50.4 | 50.6 | 98.2 |
| geometric step to zero, the recovered seed 2 (saved 0) | 99.7 | 99.7 | 99.7 | 99.7 | 99.7 | 99.7 | 98.6 | 61.3 |

Held-out band (all pairs):

| model | 0 | 1e-06 | 1e-05 | 0.0001 | 0.001 | 0.01 | 0.1 | 1 |
|---|---|---|---|---|---|---|---|---|
| anneal and hold, seed 1 (saved 0.01) | 98.3 | 98.3 | 98.3 | 98.3 | 98.2 | 97.5 | 78.2 | 49.5 |
| deep anneal, seed 1 (saved 1e-6) | 98.2 | 98.2 | 98.2 | 98.2 | 98.2 | 94.7 | 49.7 | 48.8 |
| residual dropout, 80 epochs, seed 1 (saved 1) | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 | 0.2 | 72.4 |
| residual at full scale, 40 epochs, seed 1 (saved 1) | 50.2 | 50.2 | 50.2 | 50.2 | 50.2 | 50.2 | 50.2 | 84.9 |
| geometric step to zero, the recovered seed 2 (saved 0) | 99.7 | 99.7 | 99.7 | 99.7 | 99.7 | 100.0 | 98.0 | 59.3 |

And with only the glyph rows re-scaled, numerals kept at the saved scale (held-out band):



| model | 0 | 1e-06 | 1e-05 | 0.0001 | 0.001 | 0.01 | 0.1 | 1 |
|---|---|---|---|---|---|---|---|---|
| anneal and hold, seed 1 (saved 0.01) | 50.6 | 50.6 | 50.6 | 50.9 | 53.7 | 97.5 | 76.1 | 79.4 |
| deep anneal, seed 1 (saved 1e-6) | 98.2 | 98.2 | 98.3 | 97.7 | 91.6 | 70.8 | 57.7 | 49.4 |
| residual dropout, 80 epochs, seed 1 (saved 1) | 49.5 | 49.5 | 49.5 | 49.5 | 49.5 | 49.8 | 52.0 | 72.4 |
| residual at full scale, 40 epochs, seed 1 (saved 1) | 51.0 | 51.0 | 51.0 | 51.0 | 51.0 | 51.0 | 51.9 | 84.9 |
| geometric step to zero, the recovered seed 2 (saved 0) | 99.7 | 99.7 | 99.6 | 99.3 | 97.5 | 71.2 | 50.2 | 49.4 |

What this changes:

- **The hold model already reads numerals from the ramp alone.** With its numeral residual at zero and the
  glyph rows untouched it scores 98.8 / 98.3 / 88.6 (seen / held-out / far), flat across every scale up to
  the one it was trained at. The numeral handover happened during the anneal from 1 to 0.01; nothing below
  0.01 was needed for numerals.
- **Everything between 0.01 and 1e-6 in the precision study was about the operators.** The crossing at a
  scale of 1e-4 to 1e-5, the dip-and-recovery, the cliff at exactly zero, the seed-dependent recovery and
  the failure of the learning-rate re-warm-up are the comparator losing, and sometimes relearning, the
  distinction between two token embeddings that differ by 0.3%. The deep model's achievement is to read
  that 0.005 difference, not to read numerals it could already read.
- **"Zeroing the residual gives chance" was two things.** With glyphs intact, the plain 40-epoch model with
  its numeral residual removed is at chance (50%), as claimed; the 80-epoch dropout model is not at chance
  but inverted (2.0% on seen pairs, 0.2% on the held-out band): its comparator reads the ramp with the sign
  flipped once the residual that shared the embedding with it is gone.
- **The ramp-only baseline was confounded too.** `grid_ramp` trained from scratch had no residual and
  therefore no way to tell the two operators apart, which is sufficient on its own to pin it at the coin's
  entropy. Whether the ramp alone is learnable with distinguishable operators is untested; it is being
  tested now (`--grid-numerals-only`: numerals through the encoder, operators and answers as learned rows).
  The same applies to the hot and weight-decayed ramp-only controls of M1, which were stopped.
- The residual share at the saved scale is 2.3e-2 of the generated numeral embedding for the hold model
  and 2.7e-6 for the deep model (JSON `share`), so the numeral readability floor, if the hold model is
  typical, lies at a few percent of the embedding, not at 1e-5.

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

## The clean comparison: operators as learned rows (`--grid-numerals-only`)

With the operator and answer tokens given learned embedding rows, the shared encoder and its residual
touch numerals only. The first arm to run was the decisive one: the ramp alone, from scratch, no residual
ever. Three seeds, 120 epochs, same data and schedule as the deep anneal (accuracy, %):

| arm | seen | seen gap 1 / 2-5 | held-out band | held-out both unseen (gap 1 / 2-5 / 6-20) | far band | far both unseen |
|---|---|---|---|---|---|---|
| ramp alone, operators learned (3 seeds) | 99.3 ± 0.2 | 90 / 98 | 96.0 ± 0.5 | 92 (57 / 86 / 99) | 89.8 ± 0.8 | 80 |
| anneal and hold, operators learned (seed 1; seeds 2-3 running) | 98.7 | 85 / 94 | 96.2 | 92 (55 / 90 / 100) | 78.2 | 58 |
| residual at full scale on numerals only, 40 epochs (3 seeds) | 99.0 ± 0.4 | 88 / 98 | 88.4 ± 0.7 | 77 (48 / 57 / 84) | 76.1 ± 4.1 | 55 |
| for reference: ramp alone, operators through the encoder (3 seeds) | 52.4 | chance | 49.8 | chance | 50.4 | chance |
| for reference: deep anneal to 1e-6, operators through the encoder (3 seeds) | 99.6 | 92 / 99 | 97.8 | 95 (70 / 95 / 100) | 77.7 | 64 |
| for reference: anneal and hold, operators through the encoder (3 seeds) | 99.4 | 90 / 99 | 97.5 | 95 (75 / 94 / 100) | 86.6 | 74 |

Validation loss of the ramp alone is 0.23-0.33 by epoch 10 and 0.05-0.08 at its best epoch, where the same
arm with the operators through the encoder sat at 0.693 for 120 epochs in every seed.

Reading:

- **The scaffold was never necessary.** The two-direction numeral code trains directly once the operators
  are not pushed through it. It reaches the seen precision of the 80-epoch scaffolded runs, the held-out
  band of the hold and deep arms within a point or two, and the best far band of the whole study (80% on
  pairs of two never-seen values past the trained range, against 58-74% for the annealed models and the
  93%-with-52%-held-out of the dropout arm).
- **What the residual and the fade were doing.** Every residual arm so far was compensating for an
  architecture choice: with the operator distinction in the residual, a model without one could not learn
  the task at all, a model with one could not give it up for the operators, and the fade's dynamics (the
  crossing, the cliff, the recovery) were the comparator being taught to read a 0.3% difference between
  two token embeddings. The numeral handover the hold arm showed is real but superfluous: a numeral-only
  ramp does not need to be handed anything.
- **Decision curves of the ramp alone are correct but soft.** On the psychometric split (three seeds) it is
  at 99.0-99.9% for every c, with transition widths of 21-50 values inside the held-out band and 130-152
  past the range, against 6-30 for the deep-annealed comparator at the same c. The faded comparator reads
  the ramp with more gain; the plain one reaches the same decisions with less confidence near the
  threshold. (Full table in `runs/magnitude_precision_routes/psycho/curves_numonly.json`.)
- **The residual is a genuine shortcut for numerals, not an inert passenger.** With a full-scale residual
  on numerals only (operators learned), seen precision is the same as the ramp alone (99.0%, gap 1 at 88%)
  but the held-out band drops from 96.0 to 88.4% (both-unseen pairs 92 → 77%, gaps 2-5 86 → 57%) and the
  far band from 89.8 to 76.1% (both-unseen 80 → 55%), in three seeds of three. The comparator leans on
  per-cell texture when it is there, at the expense of the shared code. (Caveat: 40 epochs against the
  ramp's 120, the precision study's convention for this arm; the ramp alone was already at 95% on the
  held-out band by its best epoch of seed 2 at 73, so the gap is not a budget effect.) So the fade was a
  cure for a self-inflicted wound: the right move is not to add the residual, and the hold arm's role
  shrinks to showing that a comparator can be weaned off one.

