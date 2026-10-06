# Independent E12 takeover audit

Reviewed inherited Cline code and revised implementation before the main run.
Reviewer: native code-reviewer, task audit_e12. Verdict: APPROVE for locked run.

Inherited blockers: correction deleted flagged rows; AUROC used maximum class
probability; rare-class CV warnings were hidden; fold-averaged macro-F1 and
undefined detection values distorted aggregation; injection documentation
promised class caps it did not implement; full CL and universal precision
ceiling claims were unsupported. Historical files are retained as provenance.

Fixed before launch: correction retains all rows; observed-label suspicion;
OOF-only scoring with singleton protection; pooled fold confusion matrices;
pooled detection counts per dataset-seed; primary inference restricted to exact
locked matrix; explicit committee replacement semantics; validation of noise
kind/rate; class floors and constant-feature NB variance; code and data hashes.

Fresh checks: check_e12.py, check_pipeline.py, compileall, end-to-end Iris CLI
smoke. Reviewer additionally checked all dual-agreement flags have identical
confident DT/NB replacement labels and class floors hold. No remaining blocker.
LSP/pyright unavailable: syntax verified with compileall. Full result review
is still required; this approval concerns execution validity, not superiority.

## Final results review — 2026-10-03

Independent reviewer recomputed the primary from raw confusion matrices:
mean +2.5205, median +2.2211, 8 positive / 2 negative datasets; exact two-sided
Wilcoxon statistic 10, p=.083984375. No overall improvement established.
All three exact protocol/domain/source/data validators passed. Main has
16,800 detection and 57,600 accuracy rows. No convergence or numerical failures;
12 expected rare-class stratification warnings remain visible. Generated
PNG/PDF/HTML checked; charts readable and match stored metrics.

Two report wording requests were fixed: list all four zero-added-noise
macro-F1 harms (glass, image segmentation, soybean, tic-tac-toe), and distinguish
relabel row retention from intentional deletion. Added a deletion invariant
check against flagged counts. Reviewer approval applies after these two fixes;
both are complete and the regenerated validator passes.
