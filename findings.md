# Current understanding

Historical E9/E5 evidence says indiscriminate hard deletion can damage
learning. Combining stages is not itself evidence of better classification.
The inherited E12 pilot cannot evaluate relabeling: its implementation
deletes the rows it intends to correct. Its AUROC score is also unsuitable.

## Candidate mechanism

NB and DT make different errors. Requiring both to confidently propose the
same alternative label might reduce false corrections while retaining rows.
This is a hypothesis, not a novelty or superiority claim. Shared errors can
still make both judges confidently wrong.

## Constraints

Original dataset labels are reference labels, not certified ground truth.
Injected corruption is known; native label noise remains unknown. Never
give corruption masks or reference training labels to a filter. Preserve
every test row. Use zero-added-noise controls and original-label references.
Do not treat folds or seeds as independent datasets for significance.

## Open questions

Does agreement trade too much recall for precision? Does correction improve
macro-F1 rather than merely detection? Does it harm small or rare classes?
Does any result generalize beyond the injected noise mechanisms?

## Completed exploratory MLP transfer

Fixed three-dataset, one-seed check: dual correction averages +1.434 macro-F1
points at 20% added noise, but diabetes loses 0.676 points; iris and vote gain.
Zero-added-noise mean is -0.162 macro-F1 points. Mixed, descriptive evidence;
no significance claim or broad deep-learning conclusion. Binary symmetric
and pairflip corruptions coincide, so they are not independent conditions.

## Completed deletion sensitivity

One seed, all ten datasets: dual-agreement deletion averages +2.829 macro-F1
points at .20 noise, but loses on breast-cancer and tic-tac-toe. DT gains are
larger than LR/NB gains. Committee deletion is descriptively stronger overall
than dual deletion. Do not cherry-pick dual as best or change the ongoing locked
correction strategy. Main three-seed inference remains pending.

## Final main evaluation (2026-10-03)

All 57,600 accuracy rows and 16,800 detection rows are present; every locked
condition completed. The fixed primary dual-correction endpoint averages
+2.520 macro-F1 points at 20% added noise across ten datasets, three classifiers
and two mechanisms. Wilcoxon p=0.0839844: overall improvement is not established
at .05. This does not prove equivalence or absence of useful effects.

Effects are conditional: final DT +5.642 macro-F1 points descriptively;
LR +0.786 and NB +1.134, with LR/NB accuracy near zero or negative.
Glass loses 1.938 macro-F1 points and tic-tac-toe loses 4.024. Eight dataset
macro-F1 aggregates are positive. Dual correction is descriptively above
committee correction (2.520 vs 2.064), but no superiority test was registered
for that comparison. Single-judge correction arms are negative at 20% noise.

Mean injected corruption is 19.889%; after dual correction reference-label
error is 16.644%. Detection precision is .640 and recall .464, averaged equally
across dataset-level results. Improved detection is insufficient by itself:
incorrect replacements and model-specific boundary changes can offset it.

With no added corruption, dual correction changes about 5.44% of reference
labels; downstream mean accuracy changes -0.094 points and macro-F1 +0.350.
This mixed average conceals harms on glass, image segmentation, soybean and
tic-tac-toe. Reference labels are unverified: do not equate these changes with
certified new noise, or call the method harmless. Protected small classes and
rare-class missing-fold support remain limitations.

## Outer-loop decision

Conclude this fixed evaluation, not the broader research question. The broad
claim is unsupported; retain conditional tree gains as a lead rather than
retune until a p-value passes. A future, separately locked study should focus
on why DT benefits, clean-label safety, and independently evaluated judge
reliability; it needs new evidence rather than post-hoc threshold selection on
these same results. No additional experiments were launched after the outcome.
