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
