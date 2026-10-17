# Identity against magnitude: can one numeral code both look up and compare?

**Status: pre-registered. One cell run on CPU as a pipeline check (below); the GPU runs are pending.**

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
