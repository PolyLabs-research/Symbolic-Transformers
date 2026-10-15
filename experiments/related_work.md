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
