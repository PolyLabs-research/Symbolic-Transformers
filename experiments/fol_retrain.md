# Retraining the FOL next-symbol model on the fixed data

The shipped checkpoint (`checkpoints/best_model.pt`) predated three fixes:

- the generator used only variable and predicate indices 0-4;
- the loss ignored every `NUM_000` target (`ignore_index=0`, though ID 0 is a numeral, not padding);
- half the test formulas (503 of 1,000) also appeared in the training set.

This note retrains the same model on the regenerated data and assesses both checkpoints against the best
achievable predictor. The new checkpoint replaces the shipped one.

All numbers come from `python scripts/assess_fol.py`. Per-target data is in `fol_retrain.json` (regenerated
test set) and `fol_shipped_original_data.json` (the shipped model on its own original data, which is
`git show 1769241^:datasets/fol_next_symbol/*.json`).

## Training

- **Configuration.** Same as the shipped run: `tiny` (d = 128, 2 layers, 4 heads, 569K parameters), batch
  32, learning rate 1e-4, 2,000 warm-up steps, seed 42. Cosine schedule over 20 epochs, which is how far
  the shipped run got.
- **Data.** `datasets/fol_next_symbol`: indices 0-99, 165K training samples from 10,000 formulas, and no
  formula shared between splits.
- **Run.** CPU, about 8 minutes per epoch after batches were trimmed to their longest context (2.6×
  faster, with identical logits). Validation loss by epoch:

  1.800, 1.743, 1.732, 1.720, 1.700, 1.653, 1.600, 1.557, 1.526, 1.514, 1.501, 1.499, 1.502, 1.495, 1.492,
  1.491, 1.492, 1.492, 1.490, **1.489** (epoch 20, kept).

  The curve sits at 1.70-1.74 for epochs 2-5, then falls by 0.22 by epoch 11. Learning to copy bound
  variables is worth about 0.21 (5.6% of targets × (ln 100 − 0.81)), so that is very likely what the drop
  is. This is not verified, because the intermediate checkpoints were pruned.
- **Replication.** The same command on an RTX 2000 Ada laptop GPU reached validation loss 1.488. On the
  test set that run scored loss 1.495, top-1 58.1%, top-1 77.2% on bound variables and 92.6% well-formed
  samples. These are within 2 points of the CPU run everywhere below.

## How the assessment works

`utils/fol_grammar.py` parses the generator's grammar. The grammar is LL(1), so a prefix fixes the parse
state, and every target gets a role:

| role | meaning |
|---|---|
| forced | only one token is grammatical |
| formula_start | the generator chooses the next (sub)formula's type |
| connective | the binary connective, uniform over four |
| arity | COMMA or RPAREN after a predicate's first argument, 50/50 |
| pred_index, quant_var, free_var | index draws, uniform over 100 values |
| bound_var | a variable reference, drawn from the variables in scope |

**The best achievable predictor is exact.** Every role's distribution is known from the generator:

- **Index draws:** uniform.
- **Bound variables:** a draw from the variables in scope.
- **Formula starts:** the generator's branching probabilities, averaged over the formula's hidden
  complexity level (1-4, weighted 0.2/0.4/0.3/0.1). Each level is weighted by how well it explains the
  formula types already seen.

On 640K formula starts from fresh generator samples, this predictor's realised loss is 1.5193 against its
expected 1.5184, and its probabilities match observed frequencies to within 0.4 points in every bin.

**Baselines.**

- **Most frequent token.**
- **Bigram counts.**
- **Longest matching suffix:** the training-set answer for the longest suffix of the context seen in
  training. For a context that occurs in training, this is the training set's own most frequent
  continuation. It is the memorisation baseline.

Results are reported overall, on contexts that never occur in the training set (68% of test targets), and
by role. Models also sample 1,000 formulas at temperature 1, scored for well-formedness and for referring
to variables in scope.

## Results on the regenerated test set

17,334 targets from 1,000 formulas, none of which occurs in training.

| method | loss | top-1 | top-5 | unseen contexts: loss / top-1 / top-5 | seen contexts: top-1 | NUM_000 targets: top-1 / top-5 |
|---|---|---|---|---|---|---|
| **best achievable** (expected) | 1.445 | 58.4 | 77.8 | 1.111 / 66.5 / 83.2 | 41.3 | 8.7 / 15.6 |
| most frequent token | n/a | 21.3 | 55.6 | n/a / 19.9 / 58.2 | 24.1 | 0.0 / 0.0 |
| bigram counts | n/a | 35.7 | 67.9 | n/a / 37.9 / 70.2 | 31.0 | 0.0 / 0.0 |
| longest matching suffix (training-set lookup) | n/a | 47.7 | 72.8 | n/a / 50.7 / 76.2 | 41.2 | 4.4 / 13.3 |
| **retrained** | 1.494 | 58.3 | 77.7 | 1.181 / 66.1 / 83.0 | 41.9 | 8.9 / 15.6 |
| shipped | 8.555 | 43.1 | 58.3 | 8.366 / 45.2 / 56.1 | 38.8 | 0.0 / 0.0 |

Top-1 accuracy (%) and loss (nats) by the role of the target:

| | forced | formula_start | connective | arity | pred_index | quant_var | free_var | bound_var |
|---|---|---|---|---|---|---|---|---|
| share of targets | 41.4 | 19.6 | 4.1 | 6.8 | 6.8 | 4.9 | 10.8 | 5.6 |
| best achievable, top-1 | 100.0 | 39.0 | 25.0 | 50.0 | 1.0 | 1.0 | 1.0 | 83.0 |
| training-set lookup, top-1 | 86.6 | 33.9 | 2.3 | 49.9 | 0.9 | 1.8 | 0.9 | 25.2 |
| **retrained, top-1** | 99.9 | 39.2 | 27.7 | 50.6 | 1.2 | 1.0 | 0.9 | 79.0 |
| shipped, top-1 | 82.0 | 39.4 | 19.1 | 2.9 | 0.6 | 0.8 | 0.8 | 5.0 |
| lowest achievable loss | 0.000 | 1.505 | 1.386 | 0.693 | 4.605 | 4.605 | 4.605 | 0.244 |
| **retrained, loss** | 0.004 | 1.527 | 1.423 | 0.696 | 4.625 | 4.648 | 4.663 | 0.806 |

Sampled formulas (1,000 at temperature 1):

| source | well-formed | bound-position variables in scope | not a training formula | distinct | mean length |
|---|---|---|---|---|---|
| generator (the test formulas) | 100.0 | 100.0 | 100.0 | 100.0 | 16.3 |
| **retrained** | 94.3 | 88.0 | 98.4 | 100.0 | 14.0 (14.6 counting malformed samples) |

Reading:

1. **The retrained model is at the ceiling of this task.**
   - Top-1 is 58.3% against a best achievable 58.4%, and top-5 77.7% against 77.8%.
   - Loss is 1.494 against a minimum of 1.445.
   - Per-role accuracies above the ceiling (connective 27.7% against 25%) are test-set noise: there are
     711 connective targets, so one standard error is 1.6 points.
2. **It is far past memorisation.**
   - It beats the training-set lookup by 10.6 points overall and by 15.4 points on contexts not in
     training.
   - On seen contexts the two are level (41.9% and 41.2%). Those are mostly short openings, where the
     next token is a random choice and the ceiling is 41.3%.
3. **It tracks variable scope.**
   - On references to bound variables it reaches 79.0% top-1 (ceiling 83.0%) and 96.5% top-5. The lookup
     gets 25.2%.
   - Doing better than the lookup requires working out from the formula which variables are bound.
4. **Its one weakness is leaked probability on bound variables.**
   - Two-thirds of the remaining 0.05 loss gap is on bound variables (0.81 against 0.24).
   - The share of probability the model puts on in-scope variables falls as more are bound:

     | distinct values in scope | probability on in-scope values (optimal 100%) | top-1 / ceiling |
     |---|---|---|
     | 1 (662 targets) | 87% | 93.8 / 100 |
     | 2 (270) | 74% | 51.9 / 50.0 |
     | 3 or more (46) | 61% | 26.1 / 32.8 |

   - Among two in-scope variables it does not favour the more recent one (49% of the time). It leaks
     probability to variables that aren't in scope.
   - This is also why 12% of variable references in sampled formulas point outside their scope, which
     the generator never does.
5. **Sampling exposes the probability tails.**
   - 94.3% of sampled formulas are well-formed. On forced tokens the model puts 99.6% of its probability
     on the right token, and sampling at temperature 1 occasionally draws from the remainder.
   - 53 of the 57 malformed samples go wrong mid-formula. Only 4 have an extra token after a complete
     formula, or end with a bracket still open.
   - Samples are slightly shorter than real formulas (14.6 against 16.3 tokens). That fits the model's
     slightly above-minimum loss on formula starts (1.527 against 1.505).
6. **It isn't memorising formulas.** 1.6% of its samples are training formulas. A fresh draw from the
   generator itself repeats a training formula 1.9-2.7% of the time (the deduplication redraw rates for
   validation and test).
7. **`NUM_000` is predicted again.** It scores 8.9% top-1 on `NUM_000` targets (best achievable 8.7%),
   against 0% for the shipped model.

## The shipped checkpoint on its own data

To see what the old numbers really were, the shipped model was assessed on its original test set. That
set has 5 indices, so index draws have a 20% ceiling, and 503 of its 1,000 formulas are also in training.

| method | loss | top-1 | top-5 | test formulas not in training: top-1 / in training: top-1 |
|---|---|---|---|---|
| best achievable (expected) | 0.775 | 62.6 | 99.0 | 62.3 / 63.6 |
| training-set lookup | n/a | 59.2 | 98.5 | 58.0 / 62.5 |
| shipped | 2.070 | 62.0 | 93.6 | 62.6 / 60.4 |

- **Its real accuracy was 62% top-1, not the README's ~44%.** That figure probably came from the old
  `evaluate_model.py`, which stripped every `NUM_000` from contexts before predicting.
- **The train/test overlap did not inflate accuracy.** The model scores lower on the overlapping formulas
  (60.4% against 62.6%), because they are short ones made mostly of random choices.
- **The `NUM_000` bug shows up in loss and top-5, not top-1.**
  - Loss is 2.07 against a minimum of 0.78.
  - Top-5 is 93.6% against 99.0%, and 0% on `NUM_000` targets.
  - The model never learned to predict index 0, which is one of the five possible indices.
- **The old task barely separated a model from a lookup table.** The lookup scored 59.2% against a ceiling
  of 62.6%. The regenerated data, with 100 indices, leaves the lookup 10.7 points short, and the model
  closes that gap.

## The benchmark suite (`symbolic_benchmarks.py`)

The retrained model passes 19 of 25 scored cases, against 18 for the shipped model, which really did fail
on large indices. None of the six remaining failures is the model's error:

| case | why it isn't a model failure |
|---|---|
| `FUNC 2` | `FUNC` never occurs in the data |
| connective expected after a complete formula | the grammar requires `MESSAGE_END` there; the model gives it 100% |
| `RPAREN` ≥ 55% inside a predicate's argument list | the true split is 50/50; the model gives 54.5% |
| two cases with an unparenthesised `AND` | that syntax never occurs in the data; the model's answers fit the grammar |
| top-5 overlap between index variants | compares tail tokens at about 0%; top-1 (`MESSAGE_END`, 100%) matches |

## Model size: Base is no better

Base (d = 512, 6 layers, 8 heads, 19.6M parameters: 34× tiny) was trained with exactly the tiny settings on
the RTX 2000 Ada laptop GPU, at about 10 minutes per epoch. Its validation loss by epoch:

1.772, 1.740, 1.748, 1.742, 1.729, 1.732, 1.706, 1.694, 1.642, 1.606, 1.574, 1.548, 1.533, 1.526, 1.518,
**1.510** (epoch 16, kept), 1.513, 1.513, 1.512, 1.512.

Its plateau lasts longer than tiny's (epochs 2-7 against 2-5) and it ends higher. By epoch 20 its training
loss (1.487) is below its validation loss (1.512): it has started to fit the 10,000 training formulas
rather than the generator. Tiny ends the other way round (training 1.503, validation 1.489, with dropout
active in training).

Test set, both models from the GPU runs:

| | tiny (569K) | base (19.6M) | best achievable |
|---|---|---|---|
| loss | 1.495 | 1.512 | 1.445 |
| top-1 / top-5 | 58.1 / 77.7 | 58.2 / 77.1 | 58.4 / 77.8 |
| unseen contexts: loss / top-1 | 1.182 / 65.8 | 1.207 / 65.9 | 1.111 / 66.5 |
| forced / formula start / connective / arity, top-1 | 99.9 / 38.8 / 25.5 / 50.8 | 99.9 / 39.0 / 26.7 / 53.3 | 100 / 39.0 / 25 / 50 |
| bound variables: top-1 / top-5 / loss | 77.2 / 96.4 / 0.825 | 74.2 / 86.3 / 1.212 | 83.0 / 100 / 0.244 |
| index draws (free variables), loss | 4.665 | 4.633 | 4.605 |
| sampled formulas: well-formed / variable references in scope | 92.6 / 86.3 | 93.4 / 81.1 | 100 / 100 |

Reading:

- **Base matches tiny on everything already at the ceiling.** Its 53.3% on arity is test-set noise around a
  fair coin: there are 1,179 arity targets, so one standard error is 1.5 points.
- **Base is worse on the one thing with room to improve.** On bound variables its top-5 drops by 10 points
  and its loss by 0.39. More of its sampled variable references point outside their scope (19% against
  14%).
- **Base's index draws are a little closer to uniform** (free-variable loss 4.633 against 4.665). Tiny
  slightly overweights some index values. This is worth 0.003 nats overall.

**At these settings, extra capacity buys nothing on this task and costs some generalisation.** Different
settings (more dropout or weight decay, or more formulas, which the generator supplies for free) might
change that. But the ceiling caps any model's possible gain over tiny at 0.1-0.3 points of top-1 (CPU and GPU
runs) and 0.05 nats of loss. Small (3.5M) was not trained.
