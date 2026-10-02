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
