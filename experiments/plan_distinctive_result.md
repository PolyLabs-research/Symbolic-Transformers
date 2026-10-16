<!-- Produced 2026-10-15 by a planning workflow: four independent proposals (generality, comparison with
existing withdrawal routes, mechanism, numeracy application), one literature audit per proposal searching for
existing work that would answer it without experiments, three judges, one synthesis. Judges' totals:
comparison 94, mechanism 92, application 78, generality 71 (two judges picked mechanism, one comparison; the
synthesis merges the two). Audit sources are in related_work.md under "Found by the shortcut audit". -->

# Plan: every withdrawal route on one task, with the readability curve as the artefact


> **Status note (2026-10-16).** M0 is done and its review found a confound that changes the premise: the
> operator and answer tokens are grid panels through the same residual as the numerals, so every arm run
> before this date faded or removed the operator distinction along with the numeral residual, and the
> ramp-only baseline could not tell the operators apart. See `withdrawal_routes.md`, section "The
> confound". M1 batch 1 was stopped (seed 1 complete for all four arms, all at chance, and one weight-decay
> seed that learned); the routes are being re-run with `--grid-numerals-only`, starting with the ramp alone
> from scratch, which is now the decisive baseline. **Result:** with learned operator rows the ramp alone
> learns directly and generalises as well as or better than every scaffolded model (99.3 / 96.0 / 89.8 on
> seen / held-out / far, three seeds). The scaffold was never necessary; this plan's premise is gone. See
> `withdrawal_routes.md`, "The clean comparison", for what remains worth running.

## 1. Goal

The artefact is one figure and its table (`experiments/withdrawal_routes.md`): every known route to a scaffold-free comparator (distillation from final and intermediate teachers, scheduled dropout, residual weight decay, stagewise removal with optimizer reset, longer or hotter ramp-only training, the step to zero, and the fade at four rates) run on the same data, seeds and model, each final model scored with the residual set to exactly zero and then re-scored across re-applied residual scales from 0 to 1. The claim under test: on this task only a fade through the readability floor gives a residual-free comparator in every seed, and the readability curve shows why the other routes fail (they never built the ramp route, or they tracked the residual down and collapsed).

## 2. Shortcut audit

Four audits, all medium confidence; arXiv and mirrors are unreachable from this machine, so most entries rest on summaries. Already answered:

- Gradual withdrawal of a discrete scaffold beats both from-scratch training and distillation on small arithmetic transformers, and collapses without optimizer reset or removal smoothing: Deng, Choi, Shieber 2024, https://arxiv.org/abs/2405.14838 (repo https://github.com/da03/Internalize_CoT_Step_by_Step); the KD route, Deng et al. 2023, https://arxiv.org/abs/2311.01460; theory for staged removal: https://arxiv.org/abs/2605.28600, https://arxiv.org/abs/2606.20937, https://arxiv.org/abs/2511.07372. So "gradual beats step and beats KD" is expected for token scaffolds; the write-up positions against these rather than claiming it.
- Both candidate floors (float absorption, Adam eps) are real and separable in a toy transformer: Prieto et al. 2025, https://arxiv.org/abs/2501.04697. The study's "below float32 readability" wording is probably 2-3 decades off (crossing at 1e-4..1e-5 against relative rounding near 1e-7; LayerNorm eps 1e-5 sits in the band) and does not go into the write-up untested.
- A fade-rate dose-response is likely predicted qualitatively by gradual-domain-adaptation bounds: https://icml.cc/virtual/2022/spotlight/18364, https://arxiv.org/abs/2002.11361.
- Final-teacher KD can fail where progressive distillation from intermediate checkpoints succeeds: Panigrahi et al. 2025, https://arxiv.org/abs/2410.05464; also https://arxiv.org/abs/2106.05945, https://arxiv.org/abs/2408.14319, https://arxiv.org/abs/1511.03643.
- Chance plateaus that never escape are often attention-softmax problems (Hoffmann et al. 2024, https://arxiv.org/abs/2310.12956), and a scalar code can learn from scratch in a small transformer (https://arxiv.org/abs/2407.15160; xVal gap interpolation, https://arxiv.org/abs/2310.02989). "The ramp alone is unlearnable" currently rests on one learning rate, one init and 120 epochs.
- Probability-scheduled withdrawal exists: https://arxiv.org/abs/1506.03099, https://arxiv.org/abs/2104.10986, https://arxiv.org/abs/2405.14853, https://arxiv.org/abs/2607.19390, US patent 11783187.

Open: no source runs any of these routes against an amplitude-faded scaffold with a scale-0 evaluation, measures the scale at which a readout abandons a faded feature, or plots accuracy against a re-applied scaffold scale. Reads owed before the write-up: Deng 2024's removal-rate ablation, Lopez-Paz Section 3, Prieto's appendix, Hermann & Lampinen 2020 (https://arxiv.org/abs/2006.12433), Grande et al. 2021 (https://arxiv.org/abs/2109.08677; the anneal-failure claim in `related_work.md` is unverified). Every link above, and the rest of what the audits found, is in `experiments/related_work.md` under "Found by the shortcut audit".

## 3. The path

**M0, this week: evaluation only** (0.5 coding day, 1 core-hour). Add `--residual-scale` to `evaluate_magnitude.py` (after line 190: `enc = model.symbol_embed.encoder`, error if `enc.residual is None`, `enc.set_residual_scale(s)`, value written to the JSON; `effective_residual()` applies the buffer after the orthogonal projection on every forward, so the override is live). Write `scripts/residual_scale_sweep.py`: every checkpoint given, scales {0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 0.1, 1}, three splits with both-unseen subsets. Run it over the existing `anneal_deep`, `anneal_hold`, `anneal_geo_long`, `anneal_geo_rewarm`, `drop`, `drop` 80-epoch, `grid_ramp_res_orth` and `sinusoid_smooth` best models. Sanity: a hold model must fall to chance at 0; deep seed 1 must give 99.0 / 98.2 / 72.3 at 1e-6 and at 0. Grafted from the application plan: a psychometric sweep (split writer for k = 0..624 against c in {375, 425, 520, 580}, `--dump-rows` on the evaluator, which already computes `p_true`). Success: hold and dropout models lose more than 30 points below 1e-4; deep models flat within 1 point. Failure means the override or point 13 is wrong, and that is fixed before anything else runs.

**M1: routes needing no trainer code** (0.5 day; 19 core-hours, about 5 h wall plus one overnight). Nine ARMS entries: `grid_ramp_lr1e3`, `grid_ramp_lr3e3`, `grid_ramp_360ep` (ramp-only controls), `grid_ramp_res_orth_wd3`, `_wd10` (shrinkage; residual norm per saved checkpoint), `anneal_deep_fade4/_fade8/_fade20` (last breakpoint 0.6 / 0.6333 / 0.7333, from the mechanism plan: the cliff as the fast end of a dose-response, and the zero-ablation dependence D(s) per checkpoint), and `sinusoid_dice` (`--sinusoid-min-period 1250`, a constant-angular-speed fixed code from scratch). Measure per run: three splits at scale 1 and 0, validation loss per epoch, residual per-cell norm and its share of the normalised embedding at the crossing (the generality audit's point: report share, not only scale). Success: no ramp-only or fixed-code run leaves 0.693; neither wd arm meets the criterion (seen >= 97%, held-out both-unseen >= 90% at scale 0) in >= 2/3 seeds; fade20 and deep recover in 3/3, fade4 in <= 1/3. If a ramp-only run learns, the scaffold claim is dead on this task and the headline becomes which setting lets the ramp learn alone; if weight decay hands over gradually, the fade is scheduled shrinkage and M2 shrinks to the distillation arms.

**M2: routes needing trainer code** (2 days; 5 arms x 3 seeds x ~45 min, 11 core-hours). Flags with unit tests: `--teacher-checkpoint` (a list, so progressive distillation walks a saved checkpoint sequence), `--distill-weight`, `--distill-temperature` (KL at the answer position, T^2 scaled; test: equals CE for a one-hot teacher at T = 1), `--residual-stage-reset N` (zero the residual in N cell-groups with Adam state reset at each stage: Deng 2024's recipe, grafted from the generality audit), `--residual-dropout-schedule START END` (relax the `[0, 1)` check at `models/transformer.py:125`, force scale 0 for selection once p = 1). Arms in priority order: `grid_ramp_kd` (hold teacher, lambda 0.9, T 2), `grid_ramp_kd_deep` (deep-fade teacher: an exactly representable function, so failure is optimisation), `grid_ramp_kd_prog` (progressive, hold teacher's `--save-every 4` sequence), `grid_ramp_res_orth_stagereset` (N = 8), `grid_ramp_res_orth_dropsched` (0.2 0.9). Measure the first epoch's time before launching. Success: none meets the criterion in >= 2/3 seeds. Whichever does becomes the primary method; M4 then adds seeds for an n = 6 comparison.

**M3: the figure** (1 day, 1 core-hour). `scripts/method_comparison_figure.py` from eval JSONs and checkpoint histories; table and exact flags in `experiments/withdrawal_routes.md`.

**M4: what the result still needs** (0.5 day, 4 core-hours). `--ln-eps 1e-2` on the deep schedule (one arm, two lines; decides whether the crossing moves with LayerNorm eps before the floor is named in prose), and seeds 4-5 for `anneal_deep` and `anneal_geo_long` so the central contrast is 5/5 versus k/5. If a competitor won M2, replace the eps arm with seeds 4-6 of the winner.

Total about 36 core-hours (9-10 h on four cores) and 4.5 coding days.

## 4. The artefact

One figure, four panels, every point a seed.

- **A, scale-0 accuracy per route.** x: routes grouped (fade x4 rates, step, step + re-warm, hold, dropout fixed/scheduled, wd3/wd10, stagewise reset, KD x3, ramp-only x4, sinusoid_dice); y: accuracy at residual scale 0 on three splits (seen; held-out both-unseen; far both-unseen); chance and criterion lines. Positive: only the fade columns (and the two lucky step seeds) sit above the criterion. Negative: any other column with two seeds above it.
- **B, validation loss against epoch** per route, the full-withdrawal epoch marked: cliff (step), dip-and-recovery (fade; steeper and failing at fade4), plateau (KD, ramp-only), equilibrium or oscillation (wd).
- **C, readability curves.** x: residual scale re-applied at evaluation, log axis 0 to 1; y: held-out both-unseen accuracy of each final model. A ramp-reading comparator is flat; one still reading the residual falls off below ~1e-4; KD and ramp-only models are flat at chance. Along-descent inset: D(s) per checkpoint for the four fade rates, with s* (the half-dependence scale) and the residual's share at s*.
- **D, psychometric step.** P(TRUE | k < c) against k for c inside the hole and past the range, deep versus hold versus sinusoid_smooth: a sharp step at a never-seen c is the readable picture.

Table columns per route and seed: seen gap 1 / 2-5, held-out both-unseen gap 1 / 2-5 / 6-20, far both-unseen, and "at 1e-6 versus exactly 0" for every faded model.

## 5. First step, this week

Code: the evaluator flag and sweep script (M0), and after the `ARMS` dict in `scripts/grid_experiment.py` (driver flags precede arm flags in `train_arm`, so an arm's `--learning-rate`/`--num-epochs` wins):

```python
ARMS["grid_ramp_lr1e3"] = ARMS["grid_ramp"] + ["--learning-rate", "1e-3"]
ARMS["grid_ramp_lr3e3"] = ARMS["grid_ramp"] + ["--learning-rate", "3e-3"]
ARMS["grid_ramp_360ep"] = ARMS["grid_ramp"] + ["--num-epochs", "360"]
ARMS["grid_ramp_res_orth_wd3"] = ARMS["grid_ramp_res_orth"] + ["--residual-wd", "3"]
ARMS["grid_ramp_res_orth_wd10"] = ARMS["grid_ramp_res_orth"] + ["--residual-wd", "10"]
_deep = ARMS["grid_ramp_res_orth_anneal_deep"][:-1]
ARMS["grid_ramp_res_orth_anneal_deep_fade4"] = _deep + ["0.2:1,0.4667:0.01,0.5667:0.01,0.6:0.000001"]
ARMS["grid_ramp_res_orth_anneal_deep_fade8"] = _deep + ["0.2:1,0.4667:0.01,0.5667:0.01,0.6333:0.000001"]
ARMS["grid_ramp_res_orth_anneal_deep_fade20"] = _deep + ["0.2:1,0.4667:0.01,0.5667:0.01,0.7333:0.000001"]
ARMS["sinusoid_dice"] = ["--embedding", "sinusoid", "--tie-weights", "--sinusoid-min-period", "1250"]
```

Runs (same generator flags and data seed as the study, so the splits are identical):

```bash
D="--out-dir runs/magnitude_precision_routes --holdout 350-449 --small-gap-frac 0.4 --small-gap-max 10 --seeds 1 2 3"
python scripts/magnitude_experiment.py $D --epochs 120 --arms grid_ramp_lr1e3 grid_ramp_lr3e3 grid_ramp_res_orth_wd3 grid_ramp_res_orth_wd10 --train-extra "--save-every 4 --keep-checkpoints 0"
python scripts/magnitude_experiment.py $D --epochs 120 --arms grid_ramp_res_orth_anneal_deep_fade4 grid_ramp_res_orth_anneal_deep_fade8 grid_ramp_res_orth_anneal_deep_fade20 sinusoid_dice --train-extra "--save-every 2 --keep-checkpoints 0"
python scripts/magnitude_experiment.py $D --epochs 120 --arms grid_ramp_360ep   # overnight
python scripts/residual_scale_sweep.py runs/magnitude_precision_anneal_v2/grid_ramp_res_orth_anneal_deep_s*/best_model.pt runs/magnitude_precision_anneal/grid_ramp_res_orth_anneal_hold_s*/best_model.pt runs/magnitude_precision_drop/grid_ramp_res_orth_drop_s*/best_model.pt --splits runs/magnitude_precision_routes/data/test*.json
```

Expected under the hypothesis: all ramp-only and `sinusoid_dice` runs at 0.693; wd arms either keep a readable residual (chance at 0) or collapse; fade20 3/3, fade4 <= 1/3; hold models cliff below 1e-4 in the sweep.

## 6. Risks and what would make us stop

- Most new columns are expected at chance, and a sceptic reads untuned baselines; the lambda = 0 control, the one-hot KD test, D1-deep's representable teacher, and progressive distillation are the answer, and a competitor winning is reported as the result, not hidden.
- Three seeds resolve only 3/3 versus 1/3; M4's extra seeds cover the one contrast that matters.
- The scaffold-necessary claim can fall in M1 (hot or long ramp-only run, or `sinusoid_dice`, learning the task). That stops the withdrawal comparison and re-plans around the ramp-only setting; it is a better headline, not a loss.
- Weight decay or stagewise reset matching the fade reduces the distinctive content to Panel C and the floor measurement; the plan proceeds with the figure either way.
- The "float32 floor" sentence may be wrong; it is replaced by the measured share at s* and the LN-eps arm, and the write-up scopes every claim to this task, model and CPU budget.
- Code touching the loss path can fake a negative; no M2 arm is reported without its control passing.
- Compute margin is thin (36 of roughly 40 core-hours); M2's last two arms are cut first if M1 overturns the plan.