# Plan: formula evaluation, a task with a right answer

**Status: Stage A run. Both formats failed every threshold, because the model never learned to look
atoms up in the world (`eval_stage_a.md`). The lookup diagnostic then showed that neither world encoding
teaches lookups to a tiny or small model (`eval_lookup.md`). Stage B is not started.**

Next-symbol prediction on generated formulas is solved: the tiny model reaches the generator's own ceiling
(`fol_retrain.md`). It tests syntax, because the formulas mean nothing. This task gives them meaning. The
model is shown a small finite world and a sentence, and must say whether the sentence is true in that
world. The answer is deterministic, so the ceiling is 100%, and a symbolic evaluator computes every label.

## The task

```
MESSAGE_START
  CONST 0 CONST 1 CONST 2                                   the domain: three objects
  SEP
  PRED 0 ( CONST 1 )  PRED 4 ( CONST 0 , CONST 2 )  ...     the true atoms; unlisted atoms are false
  MODELS
  FORALL VAR 0 ( PRED 0 ( VAR 0 ) IMPLIES EXISTS VAR 1 PRED 4 ( VAR 0 , VAR 1 ) )
  ANSWER  →  TRUE | FALSE
```

- **Every token already exists** in `unified_vocabulary.json`, including `CONST`, `MODELS` (⊨), `ANSWER`,
  `TRUE` and `FALSE`.
- **A fixed signature.** Each predicate has one arity, for example `PRED 0-2` unary and `PRED 4-5` binary.
  The current generator draws arity per atom, which a world cannot be consistent with.
- **Sentences only.** Every variable is bound. Where the generator would use a free variable, it uses a
  constant instead. Equality works between any two terms.
- **Training and assessment.** One sample per item: the context up to `ANSWER`, and the target `TRUE` or
  `FALSE`. That is the same shape as the magnitude task, so `training/train.py` needs no changes.

## Making it test evaluation, not statistics

The known failure mode for this kind of dataset is that a model learns statistical cues that happen to
predict the label within one sampling distribution. Such a model scores near 100% on its own distribution
and fails on another (Zhang et al. 2023, "On the Paradox of Learning to Reason from Data"). The design
attacks that directly:

1. **Balanced labels within strata.** TRUE and FALSE are 50/50 within every combination of outermost
   operator, depth, domain size and fact density. This uses rejection sampling, so "∃ is usually true" or
   "dense worlds make ∀ true" carry no signal.
2. **Minimal pairs.** Half the test items come with a twin world that differs in one atom and has the
   opposite answer. Pair accuracy (both right) cannot be reached without reading the world.
3. **Surface baselines that must lose:**
   - a model shown only the formula, with the world removed;
   - a model shown only the world;
   - logistic regression on bag-of-tokens and formula-skeleton features plus fact density.
4. **A sampler shift.** Generate the same task with a second sampler, with different fact density and a
   formula distribution weighted towards depth. Train on one, test on the other. A model that evaluates
   formulas does not care which sampler produced them.

## Generalisation splits

| split | train | test | what it asks |
|---|---|---|---|
| in-distribution | depth ≤ 3, 2-4 objects | same; no (world, formula) pair or formula shared with training | can it do the task at all |
| deeper formulas | depth ≤ 3 | depth 4-5 | does evaluation compose past the trained nesting |
| larger domains | 2-4 objects | 5-6 objects | do quantifiers iterate over objects never counted in training |
| more alternation | at most one ∀/∃ alternation | two | ∀∃∀ needs nested search |
| sampler shift | sampler A | sampler B | statistics vs evaluation (above) |
| renamed symbols | indices 0-49 | indices 50-99 | index invariance, cheap |

The central plot is **accuracy against evaluation cost**: the number of atom lookups a naive evaluator
makes, roughly (domain size)^(nested quantifiers) × atoms. Plot it inside and beyond the trained range.

## What would count as evidence

Thresholds to be fixed before any run:

- **Reasoning:**
  - in-distribution ≥ 95% and minimal pairs ≥ 90%;
  - on the deeper-formula and larger-domain splits, ≥ 80% where every surface baseline is ≤ 60%;
  - sampler shift within 5 points of in-distribution.
- **Statistics, not evaluation:** in-distribution high, but minimal pairs, sampler shift or the
  formula-only control near the baselines.
- **An honest middle outcome is likely.** A fixed-depth transformer may evaluate small worlds but not
  extrapolate. Quantifiers over n objects need n checks, and two layers may not be able to do that for
  n beyond training. The natural follow-up would then be a scratchpad variant, in which the model writes
  intermediate truth values before the answer. That is a separate decision.

## Build plan

1. **Check prior work first.**
   - Evans et al. 2018, "Can Neural Networks Understand Logical Entailment?": propositional entailment.
   - Clark et al. 2020, "Transformers as Soft Reasoners over Language": rule reasoning in natural language.
   - Zhang et al. 2023 (above).
   - Hahn et al. 2021, "Teaching Temporal Logics to Neural Networks".
   - Selsam et al. 2019, NeuroSAT.

   These are cited from memory and need confirming. The question is whether someone has already measured
   what this plan would measure, and which design choices they found mattered.
2. **Stage A, propositional.** Ground atoms over constants, connectives only, no quantifiers. It is a
   small, fast way to debug the generator, the evaluator and the baselines, and to check the model can
   read a world at all.
3. **Stage B, first-order.** Quantifiers, the full split set, minimal pairs and the sampler shift.
4. **Code:**
   - `data/eval_generator.py`: signature, worlds, sentences, model checker, balancing, minimal pairs and
     splits;
   - extend `utils/fol_grammar.py` for `CONST` terms;
   - `scripts/assess_eval.py`: accuracy by split and stratum, minimal pairs, the baselines and the
     cost plot;
   - tests, including a check of the evaluator against brute force on hand-built cases.
5. **Runs.** The tiny model first, about 50K training items and 2K per test split. With one target per
   item rather than one per prefix, CPU training here should be quick. Small and Base follow only if
   tiny falls short in an interesting way.

## Decisions needed

- Stage A first, or straight to first-order?
- The world encoding: an explicit domain list and listed true atoms (above), or a per-object
  description?
- Direct answers only, or plan for the scratchpad variant from the start?

## Stage A: as built, and pre-registered before any model was trained

The decisions taken were: start with Stage A, use the world encoding above, and plan the scratchpad
from the start.

- **Data.** `python data/eval_generator.py` writes `datasets/eval_stage_a` in 11 seconds. The data is
  regenerated from seed 0, and `metadata.json` records each split's sha256.
  - Signature: unary `PRED 0-3` and binary `PRED 4`.
  - Worlds: 2-4 objects named from 0-99, each possible atom true with p = 0.5, listed in random order.
  - Formulas: ground atoms under `NOT` and bracketed binary connectives.
  - Splits: 50,000 training items, with heights 1, 2 and 3 equally common. Every test split has 1,000
    minimal pairs.
- **Formats.** Both are written for every item: a direct answer, and a scratchpad. In the scratchpad,
  step 0 replaces atoms by their truth values, and each later step evaluates every innermost operator,
  so it takes one step per level.
- **Training.** `tiny` (569K parameters), batch 64, learning rate 3e-4, 500 warm-up steps, cosine
  schedule, 30 epochs, seed 1. The checkpoint kept is the one with the lowest validation loss. The
  trainer's new sequence mode puts the loss on the answer (direct) or on the whole scratchpad and
  answer.
- **Changes after the first baseline check.** The surface baselines were fitted before any training. On
  a first draft, where 44% of items had height 1, a small MLP on surface features scored 90% in
  distribution. Height-1 formulas can be answered by counting how many of their atoms are true, so the
  heights were balanced and the thresholds below are set against that baseline.

**Surface baselines** (accuracy / minimal-pair accuracy, %; they read the world but do not compose):

| | test_iid | test_iid height 3 | test_deep | test_large_world | test_shift | test_negation |
|---|---|---|---|---|---|---|
| logistic regression | 57.6 / 31.9 | 51.8 | 52.2 / 19.6 | 56.2 / 29.6 | 60.3 / 41.4 | 47.8 / 17.7 |
| MLP | 88.2 / 80.2 | 69.4 | 53.8 / 14.9 | 86.5 / 76.9 | 91.1 / 85.7 | 84.3 / 73.7 |

Any predictor that ignores the world scores 0% on pairs, because twins share their formula.

**Thresholds.** A format "evaluates formulas" in Stage A if it passes T1-T3. T4-T6 are reported either way.

| | test | threshold | best surface baseline |
|---|---|---|---|
| T1 | reads the world: test_iid pair accuracy | ≥ 90% | 80.2% |
| T2 | composes within the trained depth: test_iid accuracy at height 3 | ≥ 90% | 69.4% |
| T3 | composes past the trained depth: test_deep accuracy (heights 4-5) | ≥ 80% | 53.8% |
| T4 | larger worlds (5-6 objects): test_large_world pair accuracy | ≥ 90% | 76.9% |
| T5 | sampler shift: test_shift accuracy | within 5 points of test_iid | |
| T6 | negation of compound formulas: test_negation accuracy | within 5 points of test_iid | |

**Predictions:**

- **P1.** Both formats pass T1.
- **P2.** The direct format fails T3, scoring under 70% on test_deep. Evaluation is NC¹-complete and a
  fixed-depth model is in TC⁰, and nothing in training goes past height 3. I am unsure whether it
  passes T2.
- **P3.** The scratchpad passes T1 and T2, and beats the direct format on test_deep by at least 15
  points.
- **P4.** Where the scratchpad fails, it fails in two places:
  - positions past the longest training sequence: the deepest test traces reach 470 tokens, against
    314 in training, and absolute sinusoidal positions are not trained there;
  - step-0 lookups in larger worlds.

  `scripts/assess_eval.py` separates the two: step-0 accuracy, and accuracy by sequence length.

A confound to read with T3: deeper formulas are also longer sequences, so a failure on test_deep may be
partly a length failure. Accuracy will be reported by length within test_deep.

## Lookup diagnostic: pre-registered before it runs

Stage A failed at looking atoms up (`eval_stage_a.md`). This diagnostic isolates that step: every formula is
one atom (`python data/eval_generator.py --lookup --encoding ENC`). It compares two world encodings:

- **`true_atoms`**: the Stage A encoding, which lists the true atoms only. A lookup means detecting
  whether an atom is present, including when it is absent.
- **`full_table`**: every possible atom, each followed by TRUE or FALSE. A lookup means finding the
  matching atom and copying the next token.

Each encoding is trained with `tiny` (2 layers) and `small` (4 layers, 3.5M parameters). Settings: direct
format, batch 64, learning rate 3e-4, 500 warm-up steps, 20 epochs, seed 1. The checkpoint kept is the one
with the lowest validation loss.

- **Criterion.** Lookup counts as learned if test_iid minimal-pair accuracy is ≥ 95%.
- **test_large_world** has 5 objects, and its sequences are longer than any training item. It is
  reported, but it is not a criterion.
- **Prediction.**
  - `full_table` is learned by `tiny`.
  - `true_atoms` is not learned by `tiny`, with pair accuracy under 80%.
  - `small` on `true_atoms` is uncertain.
- **Decision rule:**
  - If only `full_table` is learned, rerun Stage A with the full-table world plus a fix for sequence
    length.
  - If both are learned, the Stage A failure was lookups inside formulas rather than the encoding, and
    that needs its own diagnosis.
  - If neither is learned, lookups need more capacity or data than these models have.

## Lookup ladder: pre-registered before it runs

The lookup diagnostic failed with both encodings (`eval_lookup.md`). Three of its four runs were flat from
epoch 1. The fourth, `true_atoms` with `tiny`, was still falling slowly at epoch 20 (validation loss
0.680 → 0.628).

The ladder keeps the `true_atoms` encoding and changes only the key a lookup has to match
(`--lookup --signature`). Every rung has about 12, 21 or 32 possible atoms per world:

| rung | signature | the key | objects |
|---|---|---|---|
| `single` | `PRED 0` only | one token, the object, at a fixed offset | 12, 21, 32 |
| `unary` | `PRED 0-3` | predicate and object, two tokens at fixed offsets | 3, 5, 8 |
| `full` | `PRED 0-3` and binary `PRED 4` (Stage A) | two or three tokens at varying offsets | 2, 3, 4 |

The `full` rung reuses `datasets/eval_lookup_true_atoms` unchanged.

- **Runs.** `tiny`, direct format, 40 epochs. The other settings are as before: batch 64, learning rate
  3e-4, 500 warm-up steps, seed 1.
- **Criterion.** A rung's lookup is learned if test_iid minimal-pair accuracy is ≥ 95%.
- **Predictions.** `single` is learned. `unary` is uncertain. `full` is not learned (pairs under 80%).
- **Decision rule:**
  - If `single` fails, even a one-token membership test is not learned in this setup. The problem is
    upstream of the key structure. Check a task known to be learnable, such as a single-token copy,
    before anything else.
  - If `single` passes and `unary` fails, two-token keys are the bottleneck. Give each atom one token
    in the world encoding.
  - If `unary` passes and `full` fails, the binary atoms are the bottleneck: three-part keys at shifted
    offsets. Give every atom the same layout.
  - If all three pass, the extra epochs were what was missing. Rerun Stage A for longer.

## Supervision-density check: pre-registered before it runs

On the ladder, the one-token rung failed (`eval_lookup.md`). Under the ladder's decision rule, a task known
to be learnable comes next. That task is associative recall: key→value pairs, then queries, which small
transformers learn when trained with many queries per sequence.

The check is a 2×2 on the `single` rung (one predicate, 12-32 objects; `--lookup --signature single`):

| | 1 query per training item | 8 queries per training item (`--queries 8`) |
|---|---|---|
| `true_atoms` | already run: 22.5% pairs | `eval_lookup_single_mq8` |
| `full_table` (associative recall) | `eval_lookup_single_table` | `eval_lookup_single_table_mq8` |

- **Multi-query items.** A training item has queries `MODELS q1 ANSWER v1 SEP q2 ANSWER v2 ...`, with loss
  on every answer. Validation and test items keep a single query, so the numbers compare directly with
  the earlier runs.
- **Runs.** `tiny`, 40 epochs. The other settings are as before: batch 64, learning rate 3e-4, 500
  warm-up steps, seed 1.
- **Criterion.** test_iid pair accuracy ≥ 95%.
- **Predictions:**
  - `full_table` with 8 queries is learned.
  - `full_table` with 1 query is not learned (pairs under 80%).
  - `true_atoms` with 8 queries is uncertain.
- **Decision rule:**
  - If `full_table` with 8 queries fails, the textbook-learnable case fails in this pipeline. Before
    anything else, the training setup itself has to be examined: the learning rate, post-LayerNorm, and
    a minimal copy task.
  - If it passes but 1 query fails, supervision density is the bottleneck. Stage A then needs many
    lookups per item, or a lookup pre-training stage.
  - If `true_atoms` with 8 queries also passes, membership is learnable with dense supervision, and
    the Stage A encoding can stay.
