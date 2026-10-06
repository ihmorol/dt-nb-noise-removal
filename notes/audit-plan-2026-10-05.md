# Supervisor readiness audit, 5 October 2026

1. Inventory current checkout, other active worktrees, experiment records, and paper versions. Establish which evidence belongs to the conference paper.
2. Audit data loading, NB likelihoods, tree parsing, filtering, selection, fold boundaries, metrics, and statistical comparisons. Run existing checks and an isolated real Weka fold.
3. Regenerate manuscript tables from recorded folds, verify numerical statements and references, rebuild the PDF, and inspect rendered pages.
4. Correct stale project documentation and record limitations, unresolved conference requirements, and verification scope.
5. Deliver the review PDF, editable source, audit findings, and a code study guide with supervisor questions.

Behavior constraint: preserve committed experiment files. Any pilot rerun uses a temporary directory or calls the fold API directly. Keep the sklearn pilot, Weka conference comparison, and later noise-correction study distinct. No accuracy-improvement or verified-noise claims without evidence.

Verification constraint: existing leakage and model checks run before code changes. Add targeted regression checks for any demonstrated defect before fixing it. Manuscript changes do not imply a full experiment rerun.
