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
