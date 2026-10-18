# Identity against magnitude: can one numeral code both look up and compare?

**Status: complete (12 GPU runs, RTX 2000 Ada).** Results follow the pre-registration.

## Why

Numerals play two roles in this project:

- **Quantities.** In the magnitude tasks, a code has to place 371 between 370 and 372. The head-to-head in
  `numeral_codes.md` made the xVal-style code the default: one learned direction scaled by the value.
- **Identities.** In the logic tasks, numerals are indices (`CONST 41`, `VAR 7`, `PRED 3`). A code then
  has to tell `CONST 41` from `CONST 42` exactly, because lookups and variable binding match keys.

The two roles pull in opposite directions:

- **xVal and the ramp are value codes of rank one and two.** For xVal, an attention score between two
  numerals is linear in the key's value, so in a single attention layer it peaks at the largest or the
  smallest key, never at the key equal to the query. The ramp has the same limit with two directions.
  Exact matching then needs the MLP to manufacture sharp features first, at a resolution of 1 in 500.
- **Count panels nest.** Panel k+1 is panel k plus one cell, so neighbours start out nearly identical.

**The grid itself offers an identity code.** Numeral k is also the k-th cell of the 25×25 grid, at row
k // 25 and column k % 25: the two base-25 digits of k. A learned row vector plus a learned column vector
identifies k. It still covers numbers never trained, as long as their row and column were. The two-channel
code adds xVal's value direction on top.

## Setup

`scripts/identity_magnitude.py`: `tiny`, the default setup (post-LN, dropout 0.1, AdamW, learning rate
3e-4, 500 warm-up steps), batch 64, 10,000 steps of fresh batches, seed 0.

- **Numbers.**
  - Training uses 0-499 minus a held-out set of 100, drawn once with seed 0. The held-out numbers are
    scattered, so every row and column of the grid still appears in training.
  - 500-624 are never seen. They are grid rows 20-24.
- **Recall.**
  - Layout: `k1 v1 ... k8 v8 ANSWER q -> v` (the layout that learned in the sanity check), values
    TRUE/FALSE, 8 queries per training sequence.
  - Half the training sequences draw 8 random keys. The other half draw 8 consecutive members of the
    training pool, so neighbouring numerals have to be told apart.
  - The plateau is about 0.64, which is answering with the majority of the listed values.
- **Compare.**
  - Layout: `a OP b ANSWER -> TRUE/FALSE`, with OP either < or >, and 8 comparisons per training sequence.
  - 40% of pairs have |a - b| ≤ 10. Chance is 0.5.
- **Evaluation.** 512 fresh one-query sequences per set, at step 10,000:
  - Recall sets:
    - `seen`: random training keys.
    - `seen_adjacent`: 8 consecutive pool members.
    - `heldout`: every key held out.
    - `far`: every key from 500-624.
  - Compare sets: `seen`, `heldout` and `far` (both numbers from the set), also broken down by gap.

**Codes.** Every other token has a learned row.

| code | numeral k is |
|---|---|
| `learned` | its own learned row |
| `xval` | s(k)·h, with s(k) = (k − 249.5) / 144.3 |
| `ramp` | its count panel through two generated directions, then LayerNorm |
| `grid_linear` | its count panel through a learned linear map |
| `grid_cell` | row vector for k // 25 plus column vector for k % 25 |
| `grid_cell_xval` | `grid_cell` plus s(k)·h |

**Runs.** The 12 runs are every code on both tasks.

## Pre-registration

- **Criteria.**
  - A recall set is learned if query accuracy is ≥ 0.95.
  - A comparison set is passed if accuracy is ≥ 0.90.
  - A code does both if recall `seen_adjacent` ≥ 0.95 and compare `heldout` ≥ 0.90.
- **Predictions, recall.**
  - `learned`, `grid_cell` and `grid_cell_xval` learn `seen` and `seen_adjacent`.
  - `xval` and `ramp` stay below 0.80 on `seen_adjacent`. On `seen`, whose keys are about 50 apart, they
    are uncertain.
  - `grid_linear` learns `seen`. It can represent a free table, because each cell's weight is free. On
    `seen_adjacent` it is uncertain.
  - `heldout`: `grid_cell` and `grid_cell_xval` learn it, because their rows and columns were trained.
    `learned` is uncertain, since its held-out rows are untrained random vectors.
  - `far`: `grid_cell` and `grid_cell_xval` stay below 0.80, because rows 20-24 were never trained.
- **Predictions, compare.**
  - `heldout`:
    - `xval`, `ramp` and `grid_cell_xval` pass.
    - `learned` is at or below 0.60.
    - `grid_cell` is uncertain. A two-digit base-25 code can be compared row first, then column, and
      every row and column was trained.
    - `grid_linear` is uncertain: training on close pairs inverted it in `magnitude_precision.md`.
  - `far`:
    - `xval` passes.
    - `grid_cell` and `learned` are at or below 0.60.
    - `ramp` and `grid_cell_xval` are uncertain. For `grid_cell_xval`, the untrained rows add noise
      to the value channel.
- **The headline prediction.** `grid_cell_xval` is the only code that does both. If `grid_cell` learns
  to compare row first, then column, it does both as well.
- **Decision rule:**
  - **`grid_cell_xval` does both and `xval` fails `seen_adjacent`:** the two-channel code becomes the
    numeral code wherever numerals are also identities, including FOL indices and lookup keys. xVal
    alone is not a safe default there.
  - **`xval` learns `seen_adjacent`:** the conflict does not show at this scale, and xVal stays the
    default.
  - **`grid_cell` alone does both:** the cell identity carries magnitude by itself for trained rows.
    The value channel then matters only past the trained range (`far`).
  - **No code does both:** report which half each code fails. Then test the two channels with a
    LayerNorm, or with a separate learned scale per channel.
- **Limits.** One seed and 10,000 steps per run. A run that is still on the plateau at 10,000 steps may
  be slow, not unable; the jump in the sanity check took 5,000 steps. The logs every 1,000 steps show
  whether a run was still moving.

## Pipeline check (CPU, before the GPU runs)

`--task recall --code learned` was run on CPU to check the script before the GPU runs. It is the same
seed, but dropout draws differ on GPU, so the GPU run is the record.

| step | 1,000-4,000 | 5,000 | 6,000-10,000 |
|---|---|---|---|
| `seen` | 0.63-0.66 | 0.982 | 1.000 |
| `seen_adjacent` | 0.63-0.66 | 1.000 | 1.000 |
| `heldout` | 0.62-0.64 | 0.686 | 0.66-0.73 |
| `far` | 0.62-0.65 | 0.729 | 0.67-0.74 |

- **The pipeline works.** The jump comes at step 5,000, as in the sanity check.
- **Learned rows handle adjacent keys without difficulty**, as predicted.
- **Untrained rows give a little, not lookup.** The held-out and far keys never had their rows trained.
  They sit just above the plateau of about 0.64, and stay there through step 10,000.

## Results (GPU)

All 12 runs used the same seed. The Python RNG that draws the data is seeded identically, so every code
saw the same training and evaluation batches. A model that gives one constant answer therefore scores
exactly the same as another constant model, and that is why several rows below repeat to the last digit.

### Recall: query accuracy at step 10,000

| code | `seen` | `seen_adjacent` | `heldout` | `far` | jump |
|---|---|---|---|---|---|
| `learned` | **1.000** | **1.000** | 0.758 | 0.744 | step 5,000 |
| `xval` | 0.922 | 0.650 | 0.902 | 0.656 | partial, from step 3,000 (`seen` 0.87-0.95 after) |
| `ramp` | 0.660 | 0.672 | 0.652 | 0.662 | none |
| `grid_linear` | 0.660 | 0.672 | 0.654 | 0.662 | none |
| `grid_cell` | **1.000** | **1.000** | **1.000** | **0.955** | step 4,000 |
| `grid_cell_xval` | **1.000** | **1.000** | **1.000** | **0.973** | step 3,000 |

### Compare: accuracy at step 10,000

| code | `seen` | `heldout` | `heldout` by gap: 1 / 2-5 / 6-20 / >20 | `far` | `far` by gap: 1 / 2-5 / 6-20 / >20 | left chance |
|---|---|---|---|---|---|---|
| `learned` | 0.484 | 0.506 | constant answer | 0.475 | constant answer | never |
| `xval` | 0.939 | **0.928** | 0.58 / 0.83 / 0.90 / 1.00 | 0.717 | 0.54 / 0.54 / 0.50 / 0.99 | step 2,000 |
| `ramp` | 0.484 | 0.506 | constant answer | 0.475 | constant answer | never |
| `grid_linear` | 0.484 | 0.506 | constant answer | 0.475 | constant answer | never |
| `grid_cell` | 0.998 | **0.992** | 1.00 / 1.00 / 0.98 / 0.99 | 0.693 | 0.86 / 0.89 / 0.63 / 0.62 | step 6,000 |
| `grid_cell_xval` | 1.000 | **1.000** | 1.00 / 1.00 / 1.00 / 1.00 | 0.730 | 1.00 / 0.89 / 0.72 / 0.63 | step 3,000 |

`xval` fluctuates from step 3,000 on: `heldout` between 0.93 and 0.99, and `far` between 0.70 and 0.91.

### Against the pre-registration

- **The headline came true, together with its stated alternative.** Both `grid_cell_xval` and
  `grid_cell` meet both criteria: `seen_adjacent` 1.000, and compare `heldout` 1.000 and 0.992. No other
  code meets either criterion on both tasks.
- **Recall predictions:**
  - **Right.**
    - `learned`, `grid_cell` and `grid_cell_xval` learn `seen` and `seen_adjacent`.
    - `xval` (0.650) and `ramp` (0.672) stay below 0.80 on `seen_adjacent`.
    - `grid_cell` and `grid_cell_xval` learn `heldout`.
  - **Wrong.**
    - "`grid_linear` learns `seen`": it never left the plateau.
    - "`grid_cell` and `grid_cell_xval` stay below 0.80 on `far`": they reached 0.955 and 0.973.
      The column vectors are trained even for 500-624. The untrained row vectors are random, but still
      distinct enough to separate keys that share a column.
  - **Open, now settled.**
    - `xval` on `seen`: 0.922, partly learned; its keys are about 50 apart.
    - `ramp` on `seen`: plateau.
    - `learned` on `heldout`: 0.758. Untrained rows give a little, not lookup.
- **Compare predictions:**
  - **Right.**
    - `xval` and `grid_cell_xval` pass `heldout`.
    - `learned` is at or below 0.60.
  - **Wrong.**
    - "`ramp` passes `heldout`": it never left chance in 10,000 steps. In `numeral_codes.md` the ramp
      needed 6-9 epochs to leave chance, so this budget may simply be too short for it.
    - "`xval` passes `far`": 0.717 at step 10,000, and 0.70-0.91 over the run.
    - "`grid_cell` at or below 0.60 on `far`": it reached 0.693.
  - **Open, now settled.**
    - `grid_cell` on `heldout`: 0.992.
    - `grid_linear` on `heldout`: chance.
    - `grid_cell_xval` on `far`: 0.730.
- **Decision rule.** Both the first and the third clause apply:
  - **First clause.** The grid-cell code becomes the numeral code wherever numerals are also
    identities. xVal alone is not a safe default there: it cannot separate adjacent keys.
  - **Third clause.** For trained rows, the cell code carries magnitude by itself, so the value channel
    could matter only past the trained range.

## Reading

- **xVal's limit with exact keys is real.** It learns lookups among keys about 50 apart, and generalises
  them to held-out keys (0.90). With 8 consecutive numbers it stays on the plateau: one scaled direction
  cannot single out k among k ± 1.
- **The grid cell is the base-625 code that does both.** Writing k as its two base-25 digits, the row
  and column of its cell, gives exact identity and ordering at once. It compares held-out numbers at gap
  1 perfectly (1.000), where xVal gets 0.58. It also left chance first on both tasks (with the value
  channel added: step 3,000 for each).
- **A caveat on that held-out result.** The held-out set is scattered, so every held-out number's row
  and column were trained. The cell code generalises by putting seen digits together in new ways, not
  by placing a number between seen numbers. A held-out band of whole rows would behave like `far`.
  For that reason the 1.000 is not comparable with the held-out band (350-449, whole rows 14-17) of
  `numeral_codes.md`.
- **Past the trained range, the combined code loses xVal's strength.** On `far` pairs with gaps above
  20, xVal alone reaches 0.99. `grid_cell_xval` reaches 0.63: it relies on the cell channel, whose rows
  20-24 were never trained. Small far gaps mostly share a row, so the trained columns decide them (gap
  1: 1.00).
  - **A natural follow-up.** Drop the cell channel at random during comparison training, so that the
    value channel has to carry magnitude on its own. The residual-dropout studies did the same for the
    per-cell residual.
- **The ramp and `grid_linear` panels learned neither task within 10,000 steps.** On comparison this
  contradicts their earlier results. Those runs used fixed datasets and more steps, so read this as
  "slow to start here", not "unable".
