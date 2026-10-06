# Supervisor meeting, 6 October 2026

## Recommendation

Take the E9f conference comparison as the coherent paper for review. Its contribution is an evaluation of how composing two DT–NB steps behaves, with fold-specific preprocessing and removal diagnostics. Do not present it as an accuracy improvement or verified label-noise detector. The later E12v2 correction experiment answers a different question and needs a separate scope decision.

## A one-minute explanation

Farid's first algorithm fits NB on training rows and discards rows whose predicted labels disagree with their supplied labels. Its second algorithm fits a tree, retains the attributes it uses, and weights each by one divided by the square root of its smallest depth. We combined these steps in both orders, ending in either NB or a tree. Every supervised stage was fitted inside each training fold; all held-out rows were scored.

Across ten datasets, plain C4.5 averaged 86.33% accuracy. The combined tree methods averaged 81.21% and 80.89%. The combined NB methods averaged 75.63% and 76.35%, while attribute weighting alone reached 77.35%. None of eight paired comparisons survived Holm adjustment. Filtering before CV produced much higher apparent scores, but it changed both label access and the cases being evaluated. That cannot prove the original paper used leakage.

## Questions you should be able to answer

- Why filter only training rows? Removing difficult test rows changes the population being measured; evaluating every held-out row tests the full prediction task.
- What does “noise” mean here? Only disagreement with NB. We have no verified erroneous labels in E9f.
- What changes between the orders? Instances-first selects attributes after deletion. Attributes-first uses a weighted judge and preserves the original training-fold tree's weights after deletion. The comparison therefore bundles order with the specified judge behavior.
- Are final tree models weighted? No. They use selected columns, not NB's likelihood weights.
- Is the weighted NB prior also weighted? No. The score is log prior plus weighted log attribute likelihoods.
- Is this nested CV? No. Learner settings are fixed; repeated stratified CV refits preprocessing inside each fold.
- Why not treat 100 folds as 100 independent observations? They reuse dataset rows. The statistical unit is each of ten datasets, using its average score.
- Does p greater than 0.05 mean the methods are equal? No. Failure to establish a difference is not an equivalence test.
- What is the biggest replication discrepancy? Our NSL-KDD tree baseline is 99.47%, versus 71.11% in the original study. Data copies, class counts, and the segmentation sample differ; exact replication remains unresolved.
- What are rare-class limitations? Some classes have fewer than ten examples, including an NSL-KDD singleton. Some folds lack a class in training or testing. Repetition cannot create more examples.
- Are category vocabularies learned only on training rows? No. File-wide vocabularies are fixed. Counts and model parameters are fitted on training rows. The estimate is conditional on known categories.
- Why include E5/E12 in the meeting but not the paper tables? Their classifiers, judging strategies, noise conditions, and evaluation questions differ. They describe project progress without strengthening claims about Algorithm 3.

## Decisions for the supervisor

Confirm the conference and its page/template/anonymity requirements, approve the authorship and affiliations, and decide whether the controlled negative-result comparison is the intended submission. The current five-page IEEE layout is a review draft until the conference requirements are known. New experiments or a new paper framing require a separate evidence plan rather than selecting favorable results from completed runs.

## Reading order today

Read the paper abstract and Results first. Then read `supervisor-code-guide.md` alongside `code/e9_farid.py`, `code/faithful_nb.py`, and `code/weka_utils.py`. Use `project-audit-2026-10-05.md` for the complete state and remaining limitations. Practice the questions above without looking at the answers.
