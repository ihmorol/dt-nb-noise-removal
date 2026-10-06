# Manuscript review and polish record

Date: 2026-10-03. Reviewed artifact: `paper/manuscript/main.tex`.

## Independent review

Two reviewers worked separately after drafting: a citation verifier given only manuscript prose and references, and a scientific/source reviewer given the manuscript and experimental artifacts. All 13 cited works exist; the verifier found no mandatory bibliographic correction or substantive misattribution. The scientific reviewer found one rounding mismatch in vote’s post-filter attribute retention: prose 7.63%, generated table 7.62%. The prose now matches the table.

## Scientific changes from the previous draft

- Replaced the implied noise-removal/improvement framing with evaluation protocol and filtering order. Judge disagreement is operational; true corruption is unverified.
- Added all 12 saved methods, including single-step attribution and parallel references, plus per-dataset macro-F1.
- Added exploratory dataset-level permutation signed-rank comparisons with Holm correction across 24 hypotheses. No adjusted p meets 0.05; no winning-order, significant-gain or equivalence claim remains.
- Described A-minus-B as a population-changing protocol contrast, not isolated leakage bias or proof of the seed paper’s protocol.
- Disclosed shared full-file vocabularies, literal missing-token categories, rare-class folds, variable macro-F1 class sets and the empty-selection fallback discrepancy.
- Made removal records visible and deduplicated totals before aggregating; complete class disappearance is reported without claiming causal identification.

## Polish and formatting

Title, abstract, introduction, related work, methods, setup, results, discussion, limitations and conclusion now form a complete draft. The normal IEEE conference paragraph settings were restored by removing custom paragraph spacing. Forced H floats, table scaling to tiny text and forced bibliography page breaks were removed. Tables use booktabs, concise consistent method names, explicit units and two separate classifier panels where appropriate. The algorithm appears in a normal figure float. Tables and numbered references are inlined into the existing source so the document is standalone for the built-in editor; generated table files remain reproducible supporting artifacts.

The final language pass preserved claim boundaries and removed unsupported technique novelty, noise-detection and performance-improvement language. Sources supporting contrary filtering outcomes are included in the research analysis so the manuscript does not become an argument against filtering in general.

## Verification and open gates

The table generator checks stored metrics against every saved fold score, coverage, class-record deduplication and a known Holm example. Source-file SHA-256 values and current analysis runtime versions are in `paper/manuscript/analysis.json`. This verifies the reanalysis, not a new full-model rerun. Source checks cover balanced braces/environments, resolved citations/references, standalone content, well-formed table rows and absence of accidental control characters.

The built-in compiler was called before and after editing and returned “Unable to find standard directories for platform.” This is a compiler/runtime diagnostic, not an identified TeX source error. The existing editor remains open. No separate PDF or replacement document was created. Successful compilation, rendered page count, overflow/overlap and float placement remain unverified, so this is not labeled submission-ready.

Before submission, choose a conference and check its page limit, anonymity, disclosure, copyright and style requirements. Keep supervisor approval of scientific framing, coauthorship and author order as an author decision. The software-version provenance comes from the saved configuration and bundled Weka JAR; the Witten book is a general tool citation, not evidence of the later exact release. An unused historical bibliography entry for multiview NB is outside the standalone manuscript’s reference set and was not used to support any claim.
