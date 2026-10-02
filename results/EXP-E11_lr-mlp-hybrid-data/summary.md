# E11 summary

- Reliable DT+NB committee filter (consensus, 6/6 cross-validated votes) replaces Farid's single-NB judge: conservative (0.2–9.9 % removed vs Alg 1's up-to-47 %), helps where data is noisy (LR +3.4 breast-cancer, +9.7 contact-lenses; MLP +2.0/+4.4), neutral on clean data — the noise-removal target is met for the cleaning stage.
- Alg 2 attribute weighting does not transfer to LR/MLP (weighted-only is the worst MLP arm, −1.0 macro-avg) — so "new data" (clean x weighted) wins only where both halves help; macro-avg new vs old: LR +0.5, MLP −1.4.
- Caveat 1: on imbalanced NSL-KDD the committee wipes rare classes (macro-F1 84.6 → 83.5) — needs a per-class removal guard (E5b).
- Caveat 2: the MLP is fragile to any perturbation (tic-tac-toe: 0.4 % deleted, clean-only −5.7); 5 seeds cap Wilcoxon p at 0.0625.
- Answer: solidify the cleaning (keep it, add the class guard), drop/re-earn the weighting copy for LR/MLP.
