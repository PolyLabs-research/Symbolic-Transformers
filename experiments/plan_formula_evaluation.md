# Plan: formula evaluation, a task with a right answer

**Status: sketch for a decision; nothing built yet.**

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
