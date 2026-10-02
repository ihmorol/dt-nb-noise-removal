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
