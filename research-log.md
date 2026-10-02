# Research takeover

2026-10-02: Preserved Cline's committed E11 and unfinished E12 in merge
9dbd29a. Earlier E9/E5 decisions retained. Inherited E12 outputs remain
historical; correction code deletes flagged rows and the ranking metric
uses maximum probability rather than observed-label suspicion.

Plan: audit first; fix metrics, class safety, and correction semantics;
compare a conservative NB+DT same-alternative agreement strategy against
each judge, consensus, uncleaned noisy data, and original-label references.
Lock the evaluation before running. Report dataset-level paired results,
including failures and zero-added-noise harms. No parameter sweeps.

2026-10-02: Protocol commit 2e9422a, implementation/protocol amendment 82919fe,
independent execution approval recorded in notes/e12v2-audit.md. Main,
deletion-sensitivity and MLP-transfer jobs launched from the same frozen code.
Smoke outputs were used only to check execution and output shape, not to tune
thresholds. Three full runs have identical recorded source hashes.

2026-10-02: Deletion and MLP runs completed with exit 0. Exact protocol, row
counts, code hashes and data hashes independently checked. Kept mixed outcomes
and harms; no hypothesis or parameter changes after reading the results. A
post-launch documentation/unused validation edit was reversed; source bytes
were restored exactly to all three recorded launch hashes.

2026-10-03: Main run completed with exit 0 after 146.12 minutes. Exact locked
matrix verified: 57,600 accuracy / 16,800 detection rows, no duplicates,
finite downstream metrics, identical code hashes and dataset fingerprints.
Main stderr contains rare-class split warnings only; no convergence, overflow,
or runtime failure. Primary gain +2.520 macro-F1, p=.0839844: not established.
Full report includes harms and competitors. Generated PDF/PNG and HTML report;
PNG visually inspected for clipping, labels and zero-baseline representation.
Outer-loop decision: conclude the bounded evaluation without outcome tuning.
