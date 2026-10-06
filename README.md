
# DT–NB hybrid research

The conference study compares NB instance filtering and tree-based attribute selection/weighting, separately and in both orders. Its main evaluation refits all supervised steps within each training fold and scores every held-out row.

### Current evidence

The authoritative conference experiment is `results/EXP-E9_farid/`: Weka J48 3.8.6, textbook `FaithfulNB`, original attribute columns, seeds 0–9, and stratified 10-fold cross-validation. The files contain twelve arms; the conference paper reports ten, excluding the later parallel comparison.

Plain C4.5 averages 86.33% accuracy. Instances-first and attributes-first tree methods average 81.21% and 80.89%. Neither combined NB method beats attribute weighting alone on mean accuracy. None of the eight post hoc accuracy comparisons is significant after Holm adjustment. These results support a controlled comparison and protocol-sensitivity study, without establishing improved classification or detection of actual label errors.

Cleaning before CV changes both label access and, for filtering arms, the evaluated sample. Its score increase cannot isolate a causal leakage effect or establish which procedure the original study used.

### Read for the supervisor meeting

- `paper/manuscript/main.tex`: current conference source.
- `paper/manuscript/conference_final.pdf`: rebuilt supervisor review PDF.
- `notes/project-audit-2026-10-05.md`: findings, evidence, remaining limits, and branch map.
- `notes/supervisor-code-guide.md`: code walkthrough, equations, examples, and questions.
- `notes/audit-artifacts/e9-validation.json`: fresh validation and result hashes.

The original README's proposed confidence-aware contribution and nested-CV/baseline promises are research plans, not completed claims of this conference study. See `ROADMAP.md` and `DECISIONS.md` for historical plans; the audit supersedes stale status summaries.

### Other research tracks

- `code/main.py`: historical sklearn CART/one-hot pipeline, distinct from the conference comparison.
- `code/replicate_farid.py`: original-column J48/NB replication checks; exact reproduction of the original scores remains unresolved.
- `code/e5_strategy.py`: later committee/soft-weight/correction treatments, outside the conference scope.
- The E12v2 study is included in this branch; see the results and protocol linked above.

### Verification

From the repository root:

```bash
python code/check_pipeline.py
python code/verify_faithful.py
python notes/audit-artifacts/verify_e9.py
python paper/manuscript/scripts/make_results_table.py
python paper/manuscript/scripts/make_conference_evidence.py
```

Python dependencies are in `requirements.txt`; Java is required for Weka. The experiment scripts write to their result directories. The E9f runner refuses to overwrite a nonempty output directory. Use `python code/e9_farid.py iris --seeds 1 --out-dir results/audit-pilot-iris` with a new directory. Other historical runners still require an isolated copy. The audit validator calls `run_fold` in a temporary directory. Full-data cleaned CSVs are inspection artifacts, not valid inputs for the main evaluation.

## Earlier branch snapshots

Alternative code, notes, and manuscript versions are preserved as ZIP files
in `archive/branch-snapshots/`. The working manuscript in
`paper/manuscript/` is the audited conference version. The previous branch
histories remain available in Git.
