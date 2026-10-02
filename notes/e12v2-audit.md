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
