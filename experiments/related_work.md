# Related work for the residual-fade study

Collected 2026-10-15 after the precision study (`magnitude_precision.md`). The question was not
novelty but whether existing work offers a shortcut to the answers we are after. Notes are from
search summaries and abstracts unless marked "read in full".

## Number representations (the embedding side is largely covered)

- xVal: a continuous numerical tokenization for scientific language models — value × a learned
  direction, continuous end to end; reports better interpolation to unseen values. This is the leading
  term of our ramp code. https://arxiv.org/html/2310.02989v2
- FoNE: precise single-token number embeddings via Fourier features — two Fourier dimensions per
  digit; beats xVal on long arithmetic; one third-party study reports weakness on multiplication and
  division. https://arxiv.org/abs/2502.09741
- Transformers can do arithmetic with the right embeddings (Abacus) — digit-position embeddings for
  length generalisation (20-digit training to 100-digit addition). https://arxiv.org/html/2405.17399
- Methods for numeracy-preserving word embeddings (DICE) — deterministic embeddings whose cosine
  distance tracks the number line. https://aclanthology.org/2020.emnlp-main.384/
- A simple yet effective number embedding approach (NEKG) — interpolates unseen numbers from
  neighbours; compares against DICE. https://aclanthology.org/2021.findings-emnlp.221.pdf
- Efficient numeracy in language models through single-token number embeddings.
  https://arxiv.org/pdf/2510.06824
- Investigating numeracy learning ability of a text-to-text transfer model — T5 interpolates
  reasonably, extrapolates badly, across magnitude, sorting, min/max. Our far-band gap is this general
  limit. https://www.researchgate.net/publication/354542695_Investigating_Numeracy_Learning_Ability_of_a_Text-to-Text_Transfer_Model
- Learning numerosity representations with transformers (OOD generalisation of counts).
  https://pmc.ncbi.nlm.nih.gov/articles/PMC8303966/

## Training-time-only features: the existing alternatives to the fade

- Unifying distillation and privileged information (Lopez-Paz, Bottou, Schölkopf, Vapnik, ICLR
  2016) — "generalized distillation": a teacher trained with the privileged feature, a student
  trained on hard labels plus the teacher's soft labels without it. The obvious existing route to a
  residual-free comparator; a direct comparison on our task is cheap. https://arxiv.org/abs/1511.03643
- Deep learning under privileged information using heteroscedastic dropout — privileged data sets
  the dropout variance rather than being withdrawn. https://arxiv.org/abs/1805.11614
- Privileged information dropout in reinforcement learning. https://arxiv.org/pdf/2005.09220
- Realistic PointGoal navigation via auxiliary losses and information bottleneck (Grande, Batra,
  Wijmans, 2021) — privileged position through an information bottleneck so the agent learns
  navigation first, localisation second. A search summary reported that annealing the bottleneck
  penalty failed to generalise because a policy learned under one regime could not adapt to the
  next, which matches our cliff finding; not verified against the paper.
  https://deepai.org/publication/realistic-pointgoal-navigation-via-auxiliary-losses-and-information-bottleneck
- Scheduled sampling (Bengio, Vinyals, Jaitly, Shazeer, NeurIPS 2015) — teacher forcing withdrawn
  on a schedule; the classic scaffold-and-fade in sequence models. Not searched this round; known.

## Curricula and noise schedules

- Curriculum Dropout (Morerio et al., ICCV 2017) — schedules the retention probability over
  training; schedules noise, not a feature. https://arxiv.org/abs/1703.06229

## Shortcut learning and shortcut removal

- Automatic shortcut removal for self-supervised representation learning (Minderer et al., ICML
  2020) — an adversarial "lens" suppresses shortcut features; keeps unmodified inputs too.
  https://proceedings.mlr.press/v119/minderer20a/minderer20a.pdf
- Localized shortcut removal (Müller et al., CVPR-W 2023) — reports oscillation: the classifier
  stops attending to the shortcut once removed, so the remover stops removing it.
  https://arxiv.org/html/2211.15510v2
- Can contrastive learning avoid shortcut solutions? (Robinson et al., NeurIPS 2021) — implicit
  feature modification removes the components currently used to discriminate, adaptively.
  https://papers.neurips.cc/paper_files/paper/2021/file/27934a1f19d678a1377c257b9a780e80-Paper-Conference.pdf
- Causally motivated shortcut removal using auxiliary labels (Makar et al., AISTATS 2022).
  https://proceedings.mlr.press/v151/makar22a/makar22a.pdf
- Preventing shortcuts in adapter training via providing the shortcuts — confounds routed through
  auxiliary modules removed at inference. https://arxiv.org/html/2510.20887v1

## What the search did not find

Searches on "feature fading", "scaffold withdrawal", "hint annealing", "input annealing" and
"privileged information annealing" returned nothing that fades a known scaffold feature below its
readability floor as the handover mechanism, nothing on a readout tracking any readable signal,
and nothing on the cliff-versus-fade distinction. Terminology in this area varies, so this is
"unlikely", not "established".

## Shortcut audit for the crossing-scale plan (2026-10-15, second pass)

Question: does existing work already say what sets the scale s* at which the comparator stops
reading the faded residual (float floor vs optimizer floor vs residual norm), or already show a
withdrawal controller driven by readability? Verdict: no direct shortcut found; 21 searches. Full
texts were not reachable from this machine (arXiv, NeurIPS, ICML and mirror DNS blocked), so entries
are from search summaries unless noted.

### Closest to the mechanism question (read these in full before M2/M3)

- Prieto, Barsbey, Mediano, Birdal 2025, "Grokking at the Edge of Numerical Stability" (ICLR 2025).
  Floating-point absorption (x + delta rounds to x) in Softmax yields exactly-zero gradient terms
  that halt learning; float64 delays the collapse, float16 hastens it, and precision "cannot be
  extended indefinitely"; a too-large Adam eps also stops learning when gradients are tiny, and they
  cite Thilak et al. that raising eps halts slingshots. Both of our candidate floors are shown to be
  real and separable in a toy transformer, but the varied quantity is logit growth, not an input
  feature's amplitude; no crossing scale is measured. https://arxiv.org/abs/2501.04697
- Thilak et al. 2022, "The Slingshot Mechanism" (NeurIPS-W 2022): late-stage Adam anomaly, cyclic
  last-layer norm growth and loss spikes; grokking without regularisation co-occurs with slingshots.
  https://arxiv.org/abs/2206.04817
- Zamirai et al. 2020, "Revisiting BFloat16 Training": nearest rounding cancels small weight
  updates in 16-bit; stochastic rounding / Kahan summation fix it. A weight-update floor, not an
  input-feature floor; relevant to interpreting any bf16/fp16 arm. https://arxiv.org/abs/2010.06192
- Adam eps as a floor when gradient RMS is tiny: NVIDIA NeMo primer
  https://docs.nvidia.com/nemo/emerging-optimizers/latest/primer/epsilon.html and Gao et al. 2024
  (sparse autoencoders; Adam loses loss-scale invariance unless eps is small)
  https://arxiv.org/abs/2406.04093
- Rounding-error analysis tools for computing the fp32 floor analytically: Beuzeville, Buttari,
  Gratton, Mary 2025 (IMA J. Numer. Anal.) https://hal-lara.archives-ouvertes.fr/NUMPEX/hal-04663142v2;
  "Limitations of neural network training due to numerical instability of backpropagation"
  https://arxiv.org/abs/2210.00805

### Fade-rate dose-response (M1) is probably predicted by existing theory

- Wang, Li, Zhao 2022, "Understanding Gradual Domain Adaptation: Improved Analysis, Optimal Path
  and Beyond" (ICML 2022): bound O(eps_0 + T*Delta + T/sqrt(n) + 1/sqrt(nT)), so an optimal number
  of steps exists and the path should minimise total length T*Delta. Frame M1 as a confirmation if
  the per-epoch shift vs margin test passes. https://icml.cc/virtual/2022/spotlight/18364
- Kumar, Ma, Liang 2020 (the exponential-in-T predecessor). https://arxiv.org/abs/2002.11361
- Scheduled-sampling schedule ablations (linear / exponential / inverse-sigmoid): exponential, which
  delays most sampling until late, works best; early fast removal hurts; no crossing point reported.
  https://arxiv.org/abs/2107.10427 and https://arxiv.org/abs/2108.12963

### Feature competition: static per run, no within-run amplitude change found

- Hermann & Lampinen 2020 (NeurIPS): availability varied across runs; the harder feature stays
  decodable but used less; no threshold scale quoted in what was reachable. Still needs a full read
  for shortcut risk 1. https://arxiv.org/abs/2006.12433
- Lampinen, Chan, Hermann 2024 (TMLR), "Learned feature representations are biased by complexity,
  learning order, position, and more": learning order and prevalence bias representations; reversing
  order by pretraining the hard feature narrows the gap. https://arxiv.org/abs/2405.05847
- Pezeshki et al. 2021, "Gradient Starvation": the fast feature starves the others of gradient
  under cross-entropy; spectral decoupling. https://arxiv.org/abs/2011.09468
- Cao, Chen, Belkin, Gu 2022, "Benign Overfitting in Two-layer CNNs": sharp SNR threshold between
  signal learning and noise memorisation; the candidate batch-noise-floor theory for s*.
  https://arxiv.org/abs/2202.06526
- Qiu et al. 2024, "Complexity Matters: dynamics of feature learning in the presence of spurious
  correlations": spurious strength and relative complexity set core-feature learning speed; core and
  spurious can be learned concurrently. https://arxiv.org/abs/2403.03375; XOR-plus-spurious phase
  analysis https://arxiv.org/abs/2606.30444; spiked-covariance NTK analysis
  https://arxiv.org/abs/2606.02830
- Murali et al., "On the Foundations of Shortcut Learning": availability and predictivity as the
  determinants, varied across runs. https://arxiv.org/abs/2310.16228
- Visual feature reliance over training (simple first, complex later): https://arxiv.org/abs/2407.06076
- Feature usage tracked across training by conditional-independence tests (not ablation): Penzel et
  al. 2022 https://pub.inf-cv.uni-jena.de/pdf/penzel2022investigating

### Within-run withdrawal of a privileged input exists in RL (probability schedules)

- Weigand, Klink, Peters, Pajarinen 2021, "Reinforcement Learning using Guided Observability"
  (PO-GRL): gradual move from full to partial observability during training.
  https://arxiv.org/abs/2104.10986
- Hu, Springer, Rybkin, Jayaraman 2024, "Privileged Sensing Scaffolds Reinforcement Learning"
  (ICLR 2024): privileged sensors in training-only components; its "Guided Observability" baseline
  drops privileged inputs with a linearly increasing probability and a 50% cutoff.
  https://arxiv.org/abs/2405.14853
- Guidance that tapers with the student's unaided success (gates, not amplitudes): "From
  Self-Distillation to Self-Practice" https://arxiv.org/abs/2609.29051; "Adaptive Teacher Exposure
  for Self-Distillation" https://arxiv.org/abs/2605.11458; Graves et al. 2017 learning-progress
  curricula https://arxiv.org/abs/1704.03003

### Step-to-zero collapse and re-warm-up

- Ash & Adams 2020 warm-starting / shrink-and-perturb https://arxiv.org/abs/1910.08475; Lee et al.
  2024 "Hare and Tortoise" https://arxiv.org/abs/2406.02596; DASH https://arxiv.org/abs/2410.23495

### Number embeddings (nothing training-only)

- Value-aware numerical representations (magnitude prefix token, kept at inference)
  https://arxiv.org/abs/2601.09706

### Not found

Nothing plots a faded feature's crossing scale against number format, optimizer eps or feature norm;
nothing anneals a simple/shortcut feature's amplitude within a run (searched "simplicity bias" +
anneal/fade/magnitude schedule, "shortcut cue strength attenuated", "spurious feature amplitude
decreased over epochs"); nothing drives scaffold withdrawal from an ablation-based readability
signal.

## Second pass (2026-10-15): shortcut audit for the numeral-codes plan

Searched again for existing work that would answer the plan's hypothesis without running it. From
search summaries only: aclanthology.org, arxiv.org and their mirrors do not resolve from this
machine, so table values below are marked unverified where the summaries did not carry them.
Verdict: no shortcut found; adjacent precedents listed.

### Numeral codes and extrapolation (the five shortcut risks)

- Wallace et al. 2019 (EMNLP). Extrapolation probes: list maximum trained on [0,150] and tested
  outside; decoding/addition figure trained on 80% of [-500,500], evaluated on [-2000,2000]. The
  paper says list-max extrapolation "falls short" of interpolation and decoding/addition fail; the
  Thawani survey adds that value embeddings "do not easily extrapolate to larger numbers, possibly
  due to instability in training". Table 7 values not retrieved. No both-unseen pair task at unit
  resolution. https://aclanthology.org/D19-1534/ (survey: https://arxiv.org/abs/2103.13136)
- DICE (Sundararaman et al., EMNLP 2020). Table 2 compares random / untrained CNN / untrained LSTM /
  value embedding / DICE-D on list max (DICE ~0.96-0.98), decoding and addition; no out-of-range
  table surfaced in any summary. https://aclanthology.org/2020.emnlp-main.384/
- xVal (Golkar et al. 2023). On the planets task the training data has no sample with a in
  (1, 1.16) and "only xVal successfully approximates these values continuously within this gap";
  OOD on planets reported as better than every other encoding. Large model and data; no
  comparison task, no unit-resolution hole, no accuracy-vs-distance curve.
  https://arxiv.org/abs/2310.02989 and https://polymathic-ai.org/blog/xval/
- NEKG (Duan, Yang, Tam, Findings EMNLP 2021): knowledge-graph embedding of number entities,
  unseen numbers interpolated from neighbours; tasks in clinical and financial domains; no
  both-unseen small-gap protocol surfaced. Code: https://github.com/hduanac/nekg
- Jiang et al. 2020, Learning Numeral Embedding (Findings EMNLP 2020): SOM/GMM prototype numerals,
  any numeral embedded as a weighted average of prototype embeddings (no per-numeral row); tasks
  are word similarity, embedding numeracy, numeral prediction, sequence labelling.
  https://arxiv.org/abs/2001.00003 and https://github.com/jeffchy/learning-numeral-embeddings
- Pal & Baral 2021 (Findings EMNLP): extrapolation = train on a lower range, test on a higher one
  (1K integers per 10K bucket over [10K, 1000K)). Numeration EM 98.06% -> 1.91% (T5-LG, secondary
  source); sorting ~75% -> ~12%; magnitude-order extrapolation numbers not retrieved.
  https://aclanthology.org/2021.findings-emnlp.265/ and https://github.com/kuntalkumarpal/T5Numeracy
- Kim et al. 2021, Have You Seen That Number? (EMNLP): QA models fail on out-of-range numbers;
  E-digit format helps. https://aclanthology.org/2021.emnlp-main.563/
- Zhou et al. 2024, Fourier features for addition (NeurIPS): from scratch GPT-2-small reaches
  53.95%, GPT-2-XL 94.44%; frozen pre-trained token embeddings give 100%; no held-out-value split
  reported in summaries. https://arxiv.org/abs/2406.03445
- FoNE (ICLR 2026): tasks are 6-digit addition, 5-digit subtraction, 3/4-digit multiplication; no
  comparison task found. https://arxiv.org/abs/2502.09741
- Yehudai et al., When Can Transformers Count to n? (UAI 2026), Appendix F.1: learned embeddings
  fail on unseen integer tokens; a continuous (1-D scalar) embedding and a random fixed mapping
  both generalise to unseen tokens on counting. The nearest published case of a scalar code
  trained from scratch generalising to unseen values in a small model; counting, not comparison.
  https://arxiv.org/abs/2407.15160
- Shultz, Nobandegani, Wang 2022/2023, A Neural Model of Number Comparison with Surprisingly
  Robust Generalization: cascade-correlation net trained on comparisons of 1-9 generalises to
  untrained 2-4 digit integers, negatives, decimals; input encoding and accuracies not retrieved.
  https://arxiv.org/abs/2210.07392
- Neural Execution Engines (NeurIPS 2020): binary number encoding interpolates numbers unseen in
  training. https://research.google/pubs/neural-execution-engines-learning-to-execute-subroutines/
- NuTime (2023): a linear layer followed by LayerNorm saturates and "fails to encode anything
  about x" far from the bias scale; motivates the constant feature in the planned linear kind.
  https://arxiv.org/abs/2310.07402
- Psychophysics of LM magnitude: Weber's Law in Transformer Magnitude Representations (2026,
  https://arxiv.org/abs/2603.20642) and Same Geometry, Opposite Noise (2026,
  https://arxiv.org/abs/2604.04469); Numeric Magnitude Comparison Effects in LLMs (Findings ACL
  2023, https://aclanthology.org/2023.findings-acl.383/). Pre-trained models, no held-out values.
- Sinusoidal structure in pre-trained number representations: https://arxiv.org/abs/2506.08966
  and https://arxiv.org/abs/2510.26285.

### Scaffold-and-fade under other names

- The Orthogonalized Read Is a Removable Training Scaffold for Recurrent Memory (Aquino-Michaels,
  2026): a read-time component that reconditions a chance-level plateau; removing it at inference
  keeps 7/9 solved models solving with a small loss, and blending it out (alpha 1 -> 0 over the
  final 30% of training) recovers near-perfect accuracy. Closest precedent for a training-only
  component faded out with a reported abrupt-vs-blended comparison; not an input feature and no
  readability floor. https://arxiv.org/abs/2607.19390
- Toward Privileged Foundation Models / PIQL (Ding & Akoglu, 2026): "teacher-annealed training"
  hands train-time-only privileged information to a module that reconstructs it from context at
  inference. https://arxiv.org/abs/2605.07799
- Self-Distilled Policy Gradient (2026): privileged-distillation weight warmed up then decayed to
  zero at the end of training to "release the student". https://arxiv.org/abs/2606.04036
- One Symptom, Three Levers (2026): gradually withholding privileged chain-of-thought so the
  student reasons alone. https://arxiv.org/abs/2608.25936; Privileged Information Distillation for
  Language Models: https://arxiv.org/abs/2602.04942
- Toward Understanding Privileged Features Distillation in Learning-to-Rank (NeurIPS 2022):
  non-monotone effect of privileged-feature predictiveness on the student.
  https://arxiv.org/abs/2209.08754; PFD at Taobao: https://arxiv.org/abs/1907.05171
- TRAM (ICML 2022): weight sharing with privileged information, marginalised at test time.
  https://proceedings.mlr.press/v162/collier22a.html
- Teacher annealing (Clark et al., BAM!, ACL 2019): distillation target moved from teacher to gold
  labels over training. https://arxiv.org/abs/1907.04829
- Annealed dropout (IBM): dropout rate annealed to zero over training.
- Learning from hints (Abu-Mostafa 1992-1995): hints as virtual examples with fixed or adaptive
  schedules; nothing on removing a hint input at inference.
  https://work.caltech.edu/paper/NIPS-1992-method-for-hints-Paper.pdf
- Modality dropout literature uses fixed rates; one paper argues against annealing the rate
  (Conformal Fusion Under Missing Modalities, https://arxiv.org/abs/2608.07183).

## Found by the shortcut audit (planning workflow, 2026-10-15)

Four auditors, one per proposed plan, each ran 6+ searches for existing work that would answer the plan without
running it. All four verdicts: no shortcut found, medium confidence (arXiv and mirrors could not be fetched from
this machine, so entries rest on search summaries). Bearing is the auditor's: "partly answers it" means read
before claiming anything nearby; "adjacent only" means related vocabulary or shape, no overlapping result.

### Audit of the generality plan (scaffold-and-fade on a second task)

- (**partly answers it**) Tsilivis, Joshi, Medvedev, Kempe, Srebro, "Learning through Internalization", arXiv 2606.20937 (June 2026) — The closest published analogue. A second set of weights operates on hints given with the input; training shifts computation onto the main weights by relying less and less on hints. Appendix MLP experiments on a 25-coordinate bipolar parity (support 12): hint coordinates are progressively replaced by random noise, stages indexed by number of hints removed; three matched-compute conditions: direct (no-hint from scratch), curriculum (sequential stages, equal budget), mixture (one stage sampled per minibatch). Provable internalization for parity with a simplified one-layer transformer as CoT tokens are progressively removed. Reports a flip side: out-of-distribution performance can degrade progressively during internalization. From search summaries only; full text not fetchable here. https://arxiv.org/abs/2606.20937
- (**partly answers it**) Deng, Choi, Shieber, "From Explicit CoT to Implicit CoT: Learning to Internalize CoT Step by Step" (Stepwise Internalization), arXiv 2405.14838 (2024) — Scaffold (CoT tokens) removed a few at a time with finetuning after each removal; GPT-2 Small reaches 99% on 9x9 multiplication (later version 20x20) where training without the scaffold fails beyond 4x4; Figure 2(a) shows the No-CoT endpoint at 0% accuracy. The recipe includes removal smoothing and an optimizer reset at each stage (ablations not confirmed from search summaries). An existing withdrawal method that is not among the plan's ten columns: discrete stagewise removal with optimizer reset, which is neither the single step nor the continuous fade. https://arxiv.org/abs/2405.14838
- (adjacent only) Huang, Zhu, Wang, Jiao, Russell, Sojoudi, Mei, "Transformers Provably Learn to Internalize Chain-of-Thought" (Log-ICoT), arXiv 2605.28600, ICLR 2026 — Removes thinking tokens in geometric chunks so k-parity is internalized in log2(k) stages with polynomial samples; a provable, geometric-in-stages withdrawal of a discrete scaffold. https://iclr.cc/virtual/2026/10016656
- (adjacent only) Bu et al., "Provable Benefit of Curriculum in Transformer Tree-Reasoning Post-Training", arXiv 2511.07372; ICML 2026 as "Provable Sample Efficiency of Curriculum Post-Training for Transformer Reasoning" — Hint-decreasing and depth-increasing curricula provably avoid an exponential sample bottleneck that direct training hits, on Countdown and parity; simulations support it. Establishes that staged hint withdrawal beats from-scratch in theory, nothing on withdrawal shape. https://proceedings.mlr.press/v306/bu26a.html
- (adjacent only) CuRIL, "When Metrics Reward the Worst Translations: Internalizing Cultural Reasoning...", arXiv 2609.08156 (2026) — Hint-injection probability decayed linearly 0.8 -> 0 over 400 GRPO steps internalizes the hint (EM 45.2); constant-probability variant never sees hint-free rollouts, collapses after ~400 steps when hints are withdrawn (EM 40.3). A probability schedule in LLM RL: the scheduled-dropout baseline the plan already includes, not a scale fade. https://arxiv.org/pdf/2609.08156
- (adjacent only) Lopez-Paz, Bottou, Scholkopf, Vapnik, "Unifying distillation and privileged information", ICLR 2016, arXiv 1511.03643 — Generalized distillation: teacher with the privileged feature, student on hard labels plus soft labels without it. No result reported for a comparable setting (a learned per-cell scaffold in a transformer embedding); it is the plan's M2 baseline and must be run to know. https://arxiv.org/abs/1511.03643
- (adjacent only) Collier et al., "Transfer and Marginalize: Explaining Away Label Noise with Privileged Information" (TRAM), ICML 2022, arXiv 2202.09244 — Privileged information used via weight sharing and approximate marginalization at test time; no annealing or withdrawal schedule. Closest hit under the privileged-information vocabulary; no schedule of any kind. https://proceedings.mlr.press/v162/collier22a.html
- (adjacent only) Grande, Batra, Wijmans, "Realistic PointGoal Navigation via Auxiliary Losses and Information Bottleneck", arXiv 2109.08677 (2021); code github.com/NicoGrande/habitat-pointnav-via-ib — Privileged ground-truth position passed through an information bottleneck with a penalty so the agent learns navigation first, localisation second. Search summaries do not mention annealing the penalty; the survey's claim that an anneal failed because the policy could not adapt across regimes remains unverified (full text not fetchable). https://arxiv.org/abs/2109.08677
- (adjacent only) Hermann and Lampinen, "What shapes feature representations? Exploring datasets, architectures, and training", NeurIPS 2020, arXiv 2006.12433 — An easy, weakly predictive feature suppresses a harder, more predictive one; as the hard feature is learned the easy one's decodability and use decline but not to chance; nonlinear decoders cannot recover the suppressed feature. Explains the study's starting observation; contains no withdrawal schedule. https://proceedings.nips.cc/paper_files/paper/2020/file/71e9c6620d381d60196ebe694840aaaa-Paper.pdf
- (adjacent only) Pezeshki et al., "Gradient Starvation: A Learning Proclivity in Neural Networks", NeurIPS 2021, arXiv 2011.09468 — Cross-entropy captures a subset of features and starves others; Spectral Decoupling (L2 on logits) lets the starved feature grow. A non-scheduled candidate fix; no report on a privileged feature withdrawn after training. https://arxiv.org/abs/2011.09468
- (adjacent only) Jolicoeur-Martineau, Lamb, Verma, Didolkar, "CNT (Conditioning on Noisy Targets)", ICML 2022, arXiv 2210.09505 — Network conditioned on a noised target as a hint, noise level random per example, pure noise at inference; framed as a curriculum from easy (low noise) to hard. A stochastic mixture over hint strengths, not a schedule to zero. https://arxiv.org/abs/2210.09505v2
- (adjacent only) Marullo, Tiezzi, Gori, Melacci, "Being Friends Instead of Adversaries: Deep Networks Learn from Data Simplified by Other Networks" (Neural Friendly Training), AAAI 2022, arXiv 2112.09968 — An auxiliary network simplifies inputs early; its influence is progressively reduced to the end of training, then dropped for deployment. A gradual fade of a training-only helper; no abrupt-versus-gradual ablation found in summaries. https://ar5iv.labs.arxiv.org/html/2112.09968
- (adjacent only) US patent 11783187, "Method, apparatus, and system for progressive training of evolving machine learning architectures" — Removes an input neuron by incrementally introducing noise into its signal while training continues, stated purpose to avoid catastrophic performance effects. Prior art for gradual input removal to avoid a cliff; no readability-floor mechanism, no measurements. https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11783187
- (adjacent only) Abu-Mostafa, "A Method for Learning from Hints", NeurIPS 1992; Romero et al., FitNets, arXiv 1412.6550 — Fixed schedules for the relative emphasis of hints (1992); FitNets' hint weighting decays over training. Schedule-shape prior art for fading a hint's influence, no cliff-versus-fade result. https://mlanthology.org/neurips/1992/abumostafa1992neurips-method
- (adjacent only) Karras et al., Progressive Growing of GANs, ICLR 2018, arXiv 1710.10196; StyleGAN-XL arXiv 2202.00273 — New layers faded in with alpha 0 -> 1 to avoid sudden shocks to trained layers; no controlled abrupt-versus-fade ablation surfaced; StyleGAN-XL notes the fade-in was later discarded for texture-sticking artifacts. Citation for the fade-in prior, not a shortcut. https://research.nvidia.com/sites/default/files/publications/karras2017gan-paper.pdf
- (adjacent only) Bengio et al., Scheduled Sampling, NeurIPS 2015, arXiv 1506.03099; Morerio et al., Curriculum Dropout, ICCV 2017, arXiv 1703.06229 — Linear, exponential and inverse-sigmoid decay of the probability that the scaffold (ground-truth token / kept unit) is present. No published abrupt-switch comparison found for scheduled sampling; a 2025 GNN surrogate paper reports linear decay of the teacher-forcing ratio reduces error accumulation versus pure teacher forcing only. https://arxiv.org/pdf/1506.03099
- (adjacent only) Braun et al., "A Curriculum Learning Method for Improved Noise Robustness in ASR" (accordion annealing), arXiv 1606.06864 — SNR annealing curriculum for recurrent speech recognizers: the signal-to-noise of the input is scheduled, not a privileged feature withdrawn. https://arxiv.org/abs/1606.06864
- (adjacent only) "Value-Aware Numerical Representations for Transformer Language Models", arXiv 2601.09706 — A learned value embedding is available at training; at decoding a learned projection of the hidden state stands in for it. The nearest number-embedding hit to a training-only numeric feature; no fade, no held-out-operand comparison task. https://arxiv.org/pdf/2601.09706
- (adjacent only) "Quiet feature learning in algorithmic tasks", AAAI 2026 — Small transformers on algorithmic tasks: validation loss flat for long stretches then drops abruptly, with features forming before the drop. Adjacent to the dip-and-recovery observation, not driven by a feature withdrawal. https://underline.io/lecture/143441-quiet-feature-learning-in-algorithmic-tasks
- (adjacent only) Grokking and weight decay: "Biologically Inspired Mechanisms for Facilitating Grokking" arXiv 2608.28184; ridge analysis arXiv 2601.19791; Kumar et al. arXiv 2310.06110 — Two-timescale account: once training loss is near zero, weight decay slowly moves parameters toward lower norm, with the delay inversely proportional to the decay strength; grokking also occurs with zero decay (feature-learning account). Background for the --residual-wd 10 baseline; nothing on a shortcut parameter group decayed to zero. https://arxiv.org/pdf/2608.28184

Open after this audit: No source found that (a) fades a feature's scale through the readout's precision floor and measures a dip-and-recovery at a fixed scale, (b) compares a single step, step plus LR restart, hold, dropout, heavy weight decay, distillation and a geometric fade side by side at matched budget, or (c) shows the same withdrawal handing over on two tasks with different readouts. The hypothesis as stated is therefore not answered by existing work and the plan should run. Three caveats. First, the sub-result the plan treats as a control, that staged withdrawal of a scaffold lets a network learn a function it cannot learn from scratch and that the mixture/dropout alternative is tested alongside, is now a crowded 2024-2026 area (Deng et al. 2405.14838; Tsilivis et al. 2606.20937; Huang et al. 2605.28600; Bu et al. 2511.07372): the write-up must position against it and the figure should gain an eleventh column, stagewise removal with optimizer reset (zero the residual cell-group by cell-group or dimension by dimension, reset Adam state at each stage), which is the published internalization recipe and is neither the step nor the fade; if it hands over in 3/3 the distinctive content shrinks to the floor mechanism. Second, full texts could not be fetched from this machine (proxy 403 on arxiv, ar5iv, alphaxiv, openreview, mlr.press, nips.cc, deepai, huggingface), so three reads are still owed before the claim is written: Tsilivis et al. (does the appendix vary stage count down to one stage, i.e. the step; does the mixture condition fail as the plan's dropout column does; what their OOD-degradation finding means for the plan's held-out improvement), Deng et al. (the removal-rate and optimizer-reset ablations, which bear directly on cliff-versus-fade and on the plan's LR re-warm result), and Grande et al. (the anneal-failure claim in experiments/related_work.md is still unverified; the code repository is public). Third, a sceptical point on the float-pinned sub-claim: a dip at residual scale 1e-4 to 1e-5 is a relative contribution of about 1e-4 to 1e-5 of the embedding, two to three decades above float32 relative resolution (1.2e-7) and at LayerNorm's eps (1e-5) and the readout gain times optimizer noise; reproducing the crossing at the same scale on task B cannot distinguish a float floor from a LayerNorm-eps or readout-gain floor unless embedding norm or eps is varied, so the plan's fallback of restating the floor as a measured residual share should be the primary measurement, not the fallback.

### Audit of the comparison plan (existing withdrawal routes head to head)

- (**partly answers it**) Deng, Choi, Shieber 2024, From Explicit CoT to Implicit CoT: Learning to Internalize CoT Step by Step (Stepwise Internalization) (listed above) — The nearest existing head-to-head of a KD route against gradual scaffold withdrawal on an arithmetic task with a tiny model. Scaffold = CoT tokens present only in training. GPT-2 Small exact-match: No CoT 0.29/0.01/0.00/0.00 at 4x4/5x5/7x7/9x9; ICoT-KD (Deng 2023 distillation incl. hidden-state matching) 0.97/0.10/-/-; ICoT-SI (remove tokens gradually, linear schedule, optimizer reset, removal smoothing) 1.00/0.95/0.95/0.99; explicit CoT 1.00. Secondary summaries of the ablation: without optimizer reset or without removal smoothing the validation curve drops to zero and never recovers. Official repo README confirms the numbers and the flags (remove_per_epoch, removal_smoothing_lambda, reset_optimizer): https://raw.githubusercontent.com/da03/Internalize_CoT_Step_by_Step/main/README.md https://arxiv.org/abs/2405.14838
- (**partly answers it**) Deng et al. 2023, Implicit Chain of Thought Reasoning via Knowledge Distillation (ICoT-KD) — The distillation route for a training-only scaffold, with hidden-state matching across layers (the analogue of arm D1b). Solves tasks that no-CoT training cannot, but on GPT-2 Small fails at 5x5 multiplication where stepwise removal succeeds (per Deng 2024 Table 2). Scaffold is discrete tokens, so no amplitude, no readability floor, no re-applied-scale diagnostic. https://arxiv.org/abs/2311.01460
- (adjacent only) Panigrahi, Liu, Malladi, Risteski, Goel, ICLR 2025, Progressive distillation induces an implicit curriculum — On sparse parity (a task hard labels learn slowly) distilling from a sequence of intermediate teacher checkpoints speeds learning and gives a provable sample-complexity benefit; the curriculum exists only in intermediate checkpoints, not in the final converged teacher. This is a stronger KD baseline than any D1 arm in the plan (final-teacher KD) and is missing from it. https://arxiv.org/abs/2410.05464
- (adjacent only) Stanton, Izmailov, Kirichenko, Alemi, Wilson, NeurIPS 2021, Does Knowledge Distillation Really Work? — Students fail to match the teacher's function even with enough capacity; the cause is optimisation, and closer teacher-matching does not always generalise better. Supports expecting D1-deep (ramp-only student of a ramp-only teacher) to fail for optimisation reasons, but says nothing about this task. https://arxiv.org/abs/2106.05945
- (adjacent only) Provodin et al., NeurIPS 2024 SciForDL workshop / TMLR 2025, Rethinking Knowledge Transfer in Learning Using Privileged Information — Re-examines Lopez-Paz et al.'s generalized distillation (only 2 of 4 synthetic experiments show a positive effect over 100 partitions) and TRAM; argues reported LUPI gains stem from dataset anomalies or architecture changes, and that state-of-the-art LUPI approaches fail to transfer knowledge from PI across domains. https://arxiv.org/abs/2408.14319
- (adjacent only) Lopez-Paz, Bottou, Scholkopf, Vapnik, ICLR 2016, Unifying distillation and privileged information (listed above) — Defines the generalized-distillation route the plan tests (teacher with PI, student on hard + temperature-softened soft labels). The 'when does it help' conditions could not be read here (arXiv and the ICLR text mirror are unreachable from this machine); the plan's intention to read Section 3 stands. https://arxiv.org/abs/1511.03643
- (adjacent only) Kim, Chin, Vasudev, Choudhury 2025, Distilling Realizable Students from Unrealizable Teachers; and DOPD 2026 ('privilege illusion') — Policy distillation under information asymmetry: multiple teacher states collapse onto one student observation, so the student can only learn the teacher's average action and degrades; fixes are selective querying/recovery states, not a schedule. DOPD (https://arxiv.org/pdf/2606.30626) argues the student can never match a teacher conditioned on information it cannot see. https://arxiv.org/abs/2505.09546
- (adjacent only) Weihs et al., NeurIPS 2021, Bridging the Imitation Gap by Adaptive Insubordination (ADVISOR) — When the teacher uses privileged information the student cannot see, imitation leaves an 'imitation gap'; a fixed sequential IL-to-RL progression fails on tasks needing frequent switching, adaptive per-step weighting works. A scheduled handoff failing where an adaptive one succeeds is the RL cousin of the cliff-vs-fade question, with no feature amplitude involved. https://arxiv.org/abs/2007.12173
- (adjacent only) Hu, Springer, Rybkin, Jayaraman, ICLR 2024, Privileged Sensing Scaffolds Reinforcement Learning (Scaffolder) — Uses the word 'scaffold' for privileged training-only sensors, but routes them only into critics, world models and reward estimators; the deployed policy never sees them, so nothing is annealed or withdrawn. No cliff/fade comparison. https://arxiv.org/abs/2405.14853
- (adjacent only) Bengio, Vinyals, Jaitly, Shazeer, NeurIPS 2015, Scheduled sampling; Huszar 2015, How (not) to train your generative model — Withdraws the ground-truth prefix by a probability annealed 1 -> 0 (the D3 analogue); reviewers found the schedule shape matters. Huszar (https://arxiv.org/abs/1511.05101) shows the objective is improper and inconsistent, and attributes its empirical success to incomplete convergence and the annealing schedule. No amplitude fade, no analysis of what the readout tracks. https://arxiv.org/abs/1506.03099
- (adjacent only) Morerio et al., ICCV 2017, Curriculum Dropout — Schedules the retention probability over training as a curriculum of increasing noise on inputs and features; image classification only, nothing removed at test, no auxiliary feature driven to p = 1. https://arxiv.org/abs/1703.06229
- (adjacent only) Lambert, Sener, Savarese, CVPR 2018, Deep Learning under Privileged Information Using Heteroscedastic Dropout — Privileged information sets the variance of Gaussian dropout on the student's activations; the PI itself is never fed to the student nor annealed. So it is not a rising-noise withdrawal, and shortcut risk 5's noise-anneal proxy arm has no precedent here. https://arxiv.org/abs/1805.11614
- (adjacent only) Grande, Batra, Wijmans 2021, Realistic PointGoal Navigation via Auxiliary Losses and Information Bottleneck (listed above) — Privileged ground-truth position passes through an information bottleneck with a usage penalty so the agent learns navigation first, localisation second; +18/21% SPL/Success semi-idealised, +15% SPL realistic. The 'annealed penalty failed to generalise' claim in related_work.md is still unverified: the paper text is unreachable from here and the repo README (github.com/NicoGrande/habitat-pointnav-via-ib, master) says nothing about a schedule. https://arxiv.org/abs/2109.08677
- (adjacent only) Hoffman, Gupta, Darrell, CVPR 2016, Learning with side information through modality hallucination; Garcia et al. 2018 adversarial modality distillation — Hidden-feature transfer of a training-only modality (depth) into an RGB-only student via Euclidean (later adversarial) matching reaches the teacher's accuracy on detection/action benchmarks. Shows hidden-state KD of privileged features can work in vision (arm D1b's mechanism); says nothing about a scaffold a student cannot learn from hard labels. https://arxiv.org/abs/1810.08437
- (adjacent only) Hoffmann et al., ICML 2024, Eureka-Moments in Transformers: Multi-Step Tasks Reveal Softmax Induced Optimization Problems — Transformers on multi-step tasks sit at a chance plateau for hundreds of epochs, some seeds never escape within 1000 epochs, and the cause is traced to the softmax in attention; proposed fixes raise the escape probability. Directly relevant to interpreting 'ramp-only never leaves chance' and to M1's lr/length sweep (longer training alone may not help; an attention-temperature fix might). https://arxiv.org/abs/2310.12956
- (adjacent only) Yehudai et al., UAI 2026 / arXiv 2024, When Can Transformers Count to n?, Appendix F.1 — Learned discrete token embeddings fail on unseen vocabulary while continuous embeddings and random mappings generalise to unseen tokens with flat absolute error. Supports the ramp code's interpolation to the held-out band; no scaffold or withdrawal. https://arxiv.org/abs/2407.15160
- (adjacent only) Liu et al., CVPR 2026, RNED (Rotary Number Encoding) supplemental, on xVal under LayerNorm — States that xVal was tested on pre-LN GPT-2 and that LayerNorm forces vector norm to sqrt(d), erasing xVal's value scaling in post-LN placements; xVal also needs an input normaliser (tanh) that saturates. Confirms the mechanism behind grid_tied's collapse and bears on the --no-embed-layernorm follow-up, not on withdrawal. https://openaccess.thecvf.com/content/CVPR2026/supplemental/Liu_RNED_Rotary_Number_CVPR_2026_supplemental.pdf
- (adjacent only) arXiv 2510.00184 (2025), Why Can't Transformers Learn Multiplication? (reverse-engineering of ICoT vs direct training) — A 2-layer transformer fine-tuned directly on 4x4 multiplication fails almost completely while the ICoT-trained (gradually de-scaffolded) model is near-perfect; the paper reverse-engineers the difference and (per secondary summary) proposes an auxiliary loss that lets direct training succeed. If an auxiliary-loss analogue exists for the ramp, the scaffold is unnecessary (M1's decision branch); secondary source only. https://www.alphaxiv.org/audio/2510.00184
- (adjacent only) arXiv 2605.28600 (2026), Transformers Provably Learn to Internalize Chain-of-Thought — Theory for gradual CoT removal: removed steps are replaced by padding, removal proceeds in geometric increments, and the argument is that the model needs parallel positions for the intermediate computation. A formal account of why gradual scaffold withdrawal works for token scaffolds; no amplitude scaffold. https://arxiv.org/pdf/2605.28600
- (adjacent only) US patent 11783187, progressive training of evolving ML architectures (listed above) — Claims removing a network input by interpolating its weights step by step toward zero, or mixing increasing noise into the input until the network ignores it, then removing it. The generic 'anneal an input to zero' idea exists as a patent claim with no experiments, no floor concept, no comparison to distillation. https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11783187

Open after this audit: No source runs any of the plan's four routes (final-teacher KD, scheduled dropout to p = 1, residual weight decay, longer/hotter ramp-only) against an amplitude-faded continuous scaffold with a residual-scale-0 evaluation on a magnitude/comparison task, so the hypothesis still has to be measured. Open items the search surfaced: (1) Deng 2024's ablation shows collapse without optimizer reset or removal smoothing, but whether it contains an explicit all-at-once (cliff) arm could not be confirmed from summaries; arXiv is unreachable from this machine, so the paper's Figure 3 and Lopez-Paz Section 3 ('when does GD help') remain unread. (2) The plan's KD arms all distill from a final teacher; Panigrahi et al. 2025 show the learnable curriculum lives in intermediate checkpoints, so a progressive-distillation arm (student follows the hold or fade teacher's checkpoint sequence) is the KD baseline most likely to overturn 'fade stands alone' and is absent from M2. (3) arXiv 2510.00184 reports an auxiliary loss that lets direct training succeed where only gradual de-scaffolding did; an analogous auxiliary target on the ramp could moot the scaffold entirely (M1's decision branch) and is cheaper than the M2 batch. (4) Hoffmann et al. 2024 attribute never-escaping chance plateaus to attention softmax, so M1's lr/epoch sweep may not be the right ramp-only control; an attention-temperature or normalisation variant would be. (5) Grande 2021's 'annealed bottleneck failed to generalise' is still unverified (README silent; code and paper unreachable here). (6) Lambert 2018 is not a rising-noise withdrawal, so the noise-anneal proxy arm has no literature precedent to match against. (7) All comparisons found use discrete token scaffolds; nothing addresses a readout tracking a signal down to the float32 floor or the readability-curve diagnostic (Panel C), which remains unique to a scalable scaffold.

### Audit of the mechanism plan (what sets the crossing scale)

- (**partly answers it**) Prieto, Barsbey, Mediano, Birdal 2025, 'Grokking at the Edge of Numerical Stability' (ICLR 2025) — Floating-point absorption (x + delta rounds to x) in Softmax produces exactly-zero gradient terms that halt learning in a small transformer; float64 delays the collapse, float16 brings it earlier, and 'precision cannot be extended indefinitely'; a too-large Adam eps also stops learning once gradients are tiny (and they cite Thilak et al. that raising eps halts slingshots). Both of the plan's candidate floors are shown to be real and separable, but the varied quantity is logit growth, not an input feature's amplitude, and no crossing scale is measured. The closest existing work; M2/M3 become a mechanism attribution for a faded input feature rather than a discovery that such floors exist. https://arxiv.org/abs/2501.04697
- (**partly answers it**) Wang, Li & Zhao 2022, 'Understanding Gradual Domain Adaptation: Improved Analysis, Optimal Path and Beyond' (ICML 2022); Kumar, Ma & Liang 2020 (ICML) — Error bound O(eps_0 + T*Delta + T/sqrt(n) + 1/sqrt(nT)) in the number of gradual steps T and per-step shift Delta, so an optimal T exists and too-few/too-large steps fail. This likely predicts M1's fade-rate dose-response qualitatively (fast fades = large per-step shift); M1 should be framed as a confirmation if the per-epoch-shift-vs-margin test passes. It says nothing about the crossing scale s* or which floor sets it. Kumar et al.: https://arxiv.org/abs/2002.11361 https://icml.cc/virtual/2022/spotlight/18364
- (adjacent only) Weigand, Klink, Peters, Pajarinen 2021, 'Reinforcement Learning using Guided Observability' (PO-GRL); Hu, Springer, Rybkin, Jayaraman 2024, 'Privileged Sensing Scaffolds Reinforcement Learning' (ICLR 2024) — Within-run withdrawal of a privileged input does exist: PO-GRL moves gradually from full to partial observability during training, and Scaffolder's 'Guided Observability' baseline drops privileged observations with a linearly increasing probability and a 50% cutoff. It is a probability schedule, not an amplitude, so it never visits the 1e-2..1e-6 band; no crossing scale, no rate ablation reported in the reachable summaries, no readability-driven controller. Scaffolder: https://arxiv.org/abs/2405.14853 https://arxiv.org/abs/2104.10986
- (adjacent only) Hermann & Lampinen 2020, 'What shapes feature representations?' (NeurIPS 2020); Lampinen, Chan & Hermann 2024, 'Learned feature representations are biased by complexity, learning order, position, and more' (TMLR) — Feature availability/easiness is varied across runs; the harder feature stays decodable but is used less; the 2024 follow-up shows learning order and prevalence bias representations and that pretraining the hard feature narrows the gap. No within-run amplitude change and no threshold scale appear in the reachable summaries; the full-text check for shortcut risk 1 is still outstanding (arXiv unreachable from this machine). 2024 paper: https://arxiv.org/abs/2405.05847 https://arxiv.org/abs/2006.12433
- (adjacent only) Cao, Chen, Belkin & Gu 2022, 'Benign Overfitting in Two-layer Convolutional Neural Networks' (NeurIPS 2022) — A sharp signal-to-noise threshold separates signal learning from noise memorisation, depending on signal strength, noise, n and d. This is the candidate 'batch-noise floor' theory for s* (a scale threshold, static per run), not a measurement of a faded feature's crossing. https://arxiv.org/abs/2202.06526
- (adjacent only) Pezeshki et al. 2021, 'Gradient Starvation' (NeurIPS 2021); Qiu et al. 2024 'Complexity Matters' (arXiv 2403.03375); XOR-plus-spurious phase analysis (arXiv 2606.30444); Murali et al. 'On the Foundations of Shortcut Learning' (arXiv 2310.16228) (listed above) — Feature competition theory: the fast/strong feature starves the others of gradient under cross-entropy; spurious strength and relative complexity set core-feature learning speed with distinct phases. All manipulate feature strength across runs and report which feature wins, not the scale at which a continuously faded feature is abandoned. https://arxiv.org/abs/2011.09468
- (adjacent only) Zamirai et al. 2020, 'Revisiting BFloat16 Training' — Nearest rounding cancels small weight updates in 16-bit training (a weight-update floor), fixed by stochastic rounding or Kahan summation. Relevant to interpreting M2's bf16/fp16 arms (distinguish activation rounding from update cancellation) but not an input-feature readability floor. https://arxiv.org/abs/2010.06192
- (adjacent only) Thilak et al. 2022, 'The Slingshot Mechanism' (NeurIPS-W 2022); NVIDIA NeMo primer on Adam epsilon; Gao et al. 2024 'Scaling and evaluating sparse autoencoders' — Late-stage Adam anomaly with cyclic last-layer norm growth and loss spikes; eps becomes a de facto hyperparameter when gradient RMS is tiny and Adam loses loss-scale invariance. Supports the plan's optimizer-floor competitor as a real mechanism and the norm-logging check, but none varies a feature's scale. Primer: https://docs.nvidia.com/nemo/emerging-optimizers/latest/primer/epsilon.html; SAE paper: https://arxiv.org/abs/2406.04093 https://arxiv.org/abs/2206.04817
- (adjacent only) Scheduled-sampling schedule ablations: Liu et al. 2021 'Confidence-Aware Scheduled Sampling' and 'Scheduled Sampling Based on Decoding Steps' — Linear vs exponential vs inverse-sigmoid decay compared; exponential (most sampling late, after the model has learned from the teacher) works best, early fast removal hurts, and a floor is sometimes needed to avoid training failure. Only end accuracy versus schedule shape; no crossing point. Second paper: https://arxiv.org/abs/2108.12963 https://arxiv.org/abs/2107.10427
- (adjacent only) Graves et al. 2017 'Automated Curriculum Learning for Neural Networks'; 'From Self-Distillation to Self-Practice' (arXiv 2609.29051); 'Adaptive Teacher Exposure for Self-Distillation' (arXiv 2605.11458) — Learning-progress-driven curricula and guidance gates that taper as the student's unaided success rises (e.g. guided share falling from 0.86 to 0.42 as pass rate rose). Closest in spirit to M4, but they gate a probability/task mix on success, not an amplitude on an ablation-based readability signal; no controller locates a crossing scale. https://arxiv.org/abs/1704.03003
- (adjacent only) Rounding-error analysis of neural networks: Beuzeville, Buttari, Gratton & Mary 2025 (IMA J. Numer. Anal.); 'Limitations of neural network training due to numerical instability of backpropagation' (arXiv 2210.00805) — Deterministic and probabilistic rounding-error bounds for forward/backward passes; the tools for computing the fp32 floor analytically (unit roundoff 2^-24 ~ 6e-8 per operation, accumulating over the first block). Supports the plan's own estimate that the observed crossing sits 1-2 decades above the float floor; no experiment on a faded feature. https://hal-lara.archives-ouvertes.fr/NUMPEX/hal-04663142v2
- (adjacent only) Ash & Adams 2020 warm-starting / shrink-and-perturb; Lee et al. 2024 'Hare and Tortoise'; DASH (arXiv 2410.23495) — Warm-started networks generalise worse after a data change; plasticity-maintenance methods help trainability more than generalisation. Relevant to the step-to-zero collapse and the re-warm-up making it worse (shortcut risk 8), separate from the crossing-scale claims. https://arxiv.org/abs/1910.08475
- (adjacent only) Value-Aware Numerical Representations for Transformer LMs (arXiv 2601.09706) — A magnitude-conditioned prefix token improves numeracy but stays present at inference; no training-only magnitude scaffold withdrawn during training was found in the number-embedding literature. https://arxiv.org/abs/2601.09706

Open after this audit: No reachable source measures the scale at which a readout abandons a continuously faded scaffold feature, varies that scale against number format / Adam eps / feature norm, or drives withdrawal from an ablation-based readability signal, so the plan's hypothesis is not already answered. Three findings change framing rather than remove the experiment: (1) Prieto et al. 2025 already shows both candidate floors (float absorption and Adam eps) are real and separable in a toy transformer, so M2/M3 should be presented as attributing a faded feature's crossing to one of two known floors, citing it; (2) Wang-Li-Zhao 2022's optimal-T bound almost certainly predicts M1's fade-rate dose-response qualitatively, so M1 is a confirmation unless the shift-vs-margin test fails; (3) PO-GRL / Scaffolder's Guided Observability show within-run privileged-input withdrawal exists in RL as a probability schedule, which should be cited as the nearest prior for the fade itself. Still unverified because every full-text route (arXiv, NeurIPS, ICML, SDSC and review mirrors) is DNS-blocked from this machine: whether Hermann & Lampinen 2020 quote any availability threshold (shortcut risk 1), whether Prieto et al.'s appendix quantifies the absorption threshold in mantissa bits (which would give the float-floor prediction line for Row 2a directly), and whether PO-GRL reports a rate ablation. Someone with arXiv access should read those three before M2 starts; the links are stored in /home/user/Symbolic-Transformers/experiments/related_work.md.

### Audit of the application plan (numeral codes, interpolation and extrapolation)

- (**partly answers it**) xVal (Golkar et al. 2023) and the Polymathic AI blog summary — A scalar-times-direction code trained end to end interpolates continuously inside a gap in the planets training data (no sample with a in (1, 1.16); only xVal approximates values in the gap) and reports better OOD than other encodings. Large model and data; no comparison task, no unit-resolution hole, no accuracy-vs-distance curve, no LayerNorm-free small-model ablation. https://arxiv.org/abs/2310.02989
- (**partly answers it**) When Can Transformers Count to n? (Yehudai et al., UAI 2026), Appendix F.1 (listed above) — In a small from-scratch transformer, learned per-token embeddings fail on unseen integer tokens while a continuous 1-D scalar embedding and a fixed random mapping both generalise to unseen tokens on a counting task. Nearest published case of a scalar code learning from scratch and generalising to unseen values; counting rather than pairwise comparison, no gap bins, no extrapolation decay curve. https://arxiv.org/abs/2407.15160
- (adjacent only) The Orthogonalized Read Is a Removable Training Scaffold for Recurrent Memory (Aquino-Michaels, 2026) — A training-only component that reconditions a chance-level plateau is removed at inference (7/9 solved models keep solving with small loss) or blended out alpha 1 -> 0 over the final 30% of training (near-perfect accuracy recovered). Direct precedent for scaffold-then-fade with an abrupt-vs-blended comparison, in recurrent memory rather than numeral codes; no readability-floor fade, no numeracy result. https://arxiv.org/abs/2607.19390
- (adjacent only) Toward Privileged Foundation Models / PIQL (Ding & Akoglu, 2026) — Train-time-only privileged information handed over by 'teacher-annealed training' to a module that reconstructs it from context at inference, in tabular foundation models. Same shape as the fade (anneal from true signal to learned substitute) but the signal is reconstructed, not driven below readability, and nothing on numbers. https://arxiv.org/abs/2605.07799
- (adjacent only) Self-Distilled Policy Gradient (2026) — Privileged-distillation weight beta warmed up then decayed to zero at the end of training (beta(k) = base * min(1,k/T_warm) * min(1,(T-k)/T_decay)) to 'release the student' from a teacher biased by inference-unavailable information. Schedule on a loss weight, not an input feature. https://arxiv.org/abs/2606.04036
- (adjacent only) Wallace et al. 2019, Do NLP Models Know Numbers? (EMNLP) — Extrapolation probes: list max trained on [0,150]; decoding/addition probe trained on 80% of [-500,500] and evaluated on [-2000,2000]. List-max extrapolation 'falls short' of interpolation; decoding/addition fail out of range; Thawani's survey adds that value embeddings 'do not easily extrapolate, possibly due to instability in training'. Table 7 values could not be retrieved (host unreachable). No both-unseen pair task at unit resolution, no decay curve per code. https://aclanthology.org/D19-1534/
- (adjacent only) DICE: Methods for Numeracy-Preserving Word Embeddings (Sundararaman et al., EMNLP 2020) — Fixed deterministic embeddings with cosine tracking the number line; Table 2 compares random, untrained CNN/LSTM, value embedding and DICE-D on list max (DICE ~0.96-0.98), decoding and addition. No out-of-range table surfaced in any summary; no both-unseen unit-resolution comparison. https://aclanthology.org/2020.emnlp-main.384/
- (adjacent only) Pal & Baral 2021, Investigating Numeracy Learning Ability of a Text-to-Text Transfer Model (Findings EMNLP) — Extrapolation = train on a lower range, test on a higher range (1K integers per 10K bucket over [10K,1000K)). Numeration EM 98.06% -> 1.91% for T5-LG (secondary source), sorting ~75% -> ~12%; magnitude-order extrapolation number not retrieved. Sets the published bar as 'collapses'; subword digits, no per-code matched comparison. https://aclanthology.org/2021.findings-emnlp.265/
- (adjacent only) NEKG: Learning Numeracy, a Simple yet Effective Number Embedding Approach Using Knowledge Graph (Duan, Yang, Tam, Findings EMNLP 2021) — Unseen numbers get embeddings interpolated from neighbouring learned rows; evaluation on clinical/financial tasks. Retains per-value parameters for neighbours; no both-unseen small-gap protocol surfaced. https://aclanthology.org/2021.findings-emnlp.221/
- (adjacent only) Learning Numeral Embedding (Jiang et al., Findings EMNLP 2020) — SOM/GMM prototype numerals; any numeral is a weighted average of prototype embeddings, so no per-numeral row is needed for OOV numerals. Tasks: word similarity, embedding numeracy, numeral prediction, sequence labelling; no unit-resolution comparison of two unseen numerals reported in summaries. https://arxiv.org/abs/2001.00003
- (adjacent only) A Neural Model of Number Comparison with Surprisingly Robust Generalization (Shultz, Nobandegani, Wang, 2022; CogSci 2023) — A cascade-correlation network trained only on comparisons of 1-9 generalises to untrained 2-, 3-, 4-digit integers, negatives and decimals and reproduces distance/ratio effects. Input encoding and accuracies not retrievable from here; not a transformer, not a token code. Worth reading in full before M3 because it is the one published comparison model with strong out-of-range generalisation. https://arxiv.org/abs/2210.07392
- (adjacent only) Pre-trained LLMs Use Fourier Features to Compute Addition (Zhou et al., NeurIPS 2024) — From scratch, GPT-2-small reaches 53.95% and GPT-2-XL 94.44% on addition; freezing pre-trained token embeddings (Fourier periods 2, 5, 10) lifts GPT-2-small to 100%. No held-out-value split reported in summaries, so it does not contradict the sinusoid_smooth memorise-and-invert result. https://arxiv.org/abs/2406.03445
- (adjacent only) FoNE: Precise Single-Token Number Embeddings via Fourier Features (ICLR 2026) — Evaluated on 6-digit addition, 5-digit subtraction, 3/4-digit multiplication; a 38M model from scratch beats fine-tuned Llama-3.2-1B; no comparison task, confirming the plan's note. https://arxiv.org/abs/2502.09741
- (adjacent only) NuTime: Numerically Multi-Scaled Embedding (2023) — Analyses a linear layer followed by LayerNorm for a scalar: the output saturates and 'fails to encode anything about x' far from the bias scale. Supports the plan's risk that the xVal-style arm under embedding LayerNorm needs a constant feature or a no-LN variant. https://arxiv.org/abs/2310.07402
- (adjacent only) Weber's Law in Transformer Magnitude Representations (2026) and Same Geometry, Opposite Noise (2026) — Psychophysics framing (distance effect, Weber fraction, scalar variability) for magnitude representations in pre-trained language models. Adjacent to the Panel C psychometric sweep; they probe pre-trained models on seen numbers, not never-seen values in a hole or past the range. https://arxiv.org/abs/2603.20642
- (adjacent only) Toward Understanding Privileged Features Distillation in Learning-to-Rank (Yang et al., NeurIPS 2022) — Teacher with privileged features, student without; student performance rises then falls as the privileged feature becomes more predictive. Relevant to the generalised-distillation route named in the plan, not to the fade. https://arxiv.org/abs/2209.08754
- (adjacent only) BAM! Born-Again Multi-Task Networks (Clark et al., ACL 2019), 'teacher annealing' — Distillation target annealed from teacher predictions to gold labels over training. The established use of the term 'annealing' in distillation; it anneals a target, not an input feature. https://arxiv.org/abs/1907.04829
- (adjacent only) Learning from hints (Abu-Mostafa, NeurIPS 1992; Neural Computation 1995) — Hints as virtual examples with fixed or adaptive schedules of emphasis; nothing about a hint input withdrawn at inference or faded below readability. https://work.caltech.edu/paper/NIPS-1992-method-for-hints-Paper.pdf
- (adjacent only) Conformal Fusion Under Missing Modalities (2026) and modality-dropout literature — Modality dropout is used at fixed rates; this paper argues explicitly against annealing the drop rate toward the test-time rate. No annealed-to-zero auxiliary channel with removal at test time was found. https://arxiv.org/abs/2608.07183

Open after this audit: No source was found that reads a numeral code with zero per-value parameters at unit resolution on both-unseen pairs, or that publishes an extrapolation decay curve per code under matched training, so the plan's two-part hypothesis is not answered by existing work. Two items could still weaken it and should be read in full (tables unreachable from this machine: aclanthology.org, arxiv.org and all mirrors fail DNS; only github.com resolves): (1) Wallace et al. 2019 Table 7 and DICE's experiments section, to confirm no fixed code already extrapolates well on an ordering probe; (2) Shultz et al. 2022, whose cascade-correlation comparison model generalises from 1-9 to 4-digit integers, to see what input code gives it that reach. The closest mechanism precedents (Orthogonalized Read scaffold blended out over the last 30% of training, PIQL teacher-annealed privileged information, SDPG decayed distillation weight) show that 'training-only component faded out' is now a recognised move in 2026 work, so the write-up should cite them and claim the readability-floor fade and the readout tracking as the specific contribution rather than the fade itself. The xVal planets-gap interpolation and the Count-to-n scalar-embedding result mean shortcut risk 2 (a scalar code may learn from scratch in a small model) is live and M2's linear arm with and without embedding LayerNorm must be run before claiming the scaffold is necessary. Links from this audit were appended to /home/user/Symbolic-Transformers/experiments/related_work.md.

## Formula evaluation (checked 2026-10-16, before building Stage A)

Searched for work that already answers what `plan_formula_evaluation.md` would measure. The notes come
from search summaries and abstracts. arxiv.org, export.arxiv.org and the Semantic Scholar API are blocked
from this machine, so no full text was read.

- **Theory: formula evaluation needs serial steps.** Boolean formula evaluation is NC¹-complete
  (Buss 1987), while fixed-depth transformers without chain of thought sit in TC⁰. So unless TC⁰ = NC¹,
  no fixed-depth model evaluates formulas of unbounded depth; with O(n) chain-of-thought steps,
  constant-depth transformers can (Li, Liu, Zhou & Ma, "Chain of Thought Empowers Transformers to Solve
  Inherently Serial Problems", ICML 2023). Feng et al. 2023 ("Towards Revealing the Mystery behind Chain
  of Thought") make the same point for arithmetic expressions. **Consequence for Stage A:** a direct
  answer should not extrapolate to deeper formulas, while a scratchpad of local steps can.
  https://icml.cc/virtual/2023/27237 https://arxiv.org/pdf/2305.15408
- **Zhang, Li, Meng, Chang & Van den Broeck, "On the Paradox of Learning to Reason from Data"** (IJCAI
  2023). BERT reaches near-perfect accuracy on its own propositional-deduction distribution
  (SimpleLogic) and fails on others, because it learns statistical features inherent to such problems.
  Removing those features jointly is infeasible, though BERT has the capacity to solve the task exactly.
  **Consequence:** minimal pairs, a sampler shift and surface baselines.
  https://arxiv.org/abs/2205.11502 https://ijcai.org/proceedings/2023/375
- **Tong, Çakar & Pehlevan, "Boule or Baguette? A Study on Task Topology, Length Generalization, and the
  Benefit of Reasoning Traces"** (February 2026). PITA has 23M propositional statements with proofs.
  Models trained with reasoning traces generalise well on "broad, shallow" subsets and worse than
  direct prediction on "narrow, deep" ones. **Consequence:** keep the trace short, one step per formula
  level rather than one per operator. https://arxiv.org/abs/2602.14404
- **Langedijk, Jumelet & Zuidema, "Propositional Logic for Probing Generalization in Neural Networks"**
  (2025; later titled "Logic Satisfiability as a Probe for Inductive Biases in Neural Networks").
  Transformers, GCNs and LSTMs generate satisfying assignments. All do well in distribution, but
  transformers do not apply negation compositionally without added structural bias. **Consequence:**
  the `test_negation` split. https://arxiv.org/abs/2506.08978
- **Lazić, Fowl, György & Szepesvári, "To See the Unseen: on the Generalization Ability of Transformers
  in Symbolic Reasoning"** (April 2026). Output vectors for tokens unseen in training collapse together,
  which explains why variable names unseen in training fail. Remedies include an easier copy
  mechanism, more data diversity, and resetting or freezing embeddings. **Consequence:** object names
  are drawn from the trained range for every world, not held out. https://arxiv.org/abs/2604.21632
- **Hahn, Schmitt, Kreber, Rabe & Finkbeiner, "Teaching Temporal Logics to Neural Networks"** (ICLR
  2021; also "Transformers Generalize to the Semantics of Logics"). Transformers predict satisfying
  assignments and generalise to larger formulas than trained. Tree positional encodings help: 86.1%
  against 75.7% semantic accuracy at size 35-50. https://arxiv.org/abs/2003.04218
- **Evans, Saxton, Amos, Kohli & Grefenstette, "Can Neural Networks Understand Logical Entailment?"**
  (ICLR 2018). On propositional entailment, tree-structured networks beat LSTMs, which beat CNNs, and
  PossibleWorldNets beat all of them. https://arxiv.org/abs/1802.08535

Nothing found measures the plan's combination: a tiny model evaluating formulas against explicit finite
worlds, with minimal pairs, direct answers and a level-wise scratchpad compared across depth and world
size. The theory predicts the direct format's failure past the trained depth, so that result alone would
confirm known theory, not be new.
