# DT–NB hybrid research

The conference study compares NB instance filtering and tree-based attribute selection/weighting, separately and in both orders. Its main evaluation refits all supervised steps within each training fold and scores every held-out row.

### Current evidence

The authoritative conference experiment is `results/EXP-E9_farid/`: Weka J48 3.8.6, textbook `FaithfulNB`, original attribute columns, seeds 0–9, and stratified 10-fold cross-validation. The files contain twelve arms; the conference paper reports ten, excluding the later parallel comparison.

Plain C4.5 averages 86.33% accuracy. Instances-first and attributes-first tree methods average 81.21% and 80.89%. Neither combined NB method beats attribute weighting alone on mean accuracy. None of the eight post hoc accuracy comparisons is significant after Holm adjustment. These results support a controlled comparison and protocol-sensitivity study, without establishing improved classification or detection of actual label errors.

Cleaning before CV changes both label access and, for filtering arms, the evaluated sample. Its score increase cannot isolate a causal leakage effect or establish which procedure the original study used.

### Read for the supervisor meeting

- `paper/manuscript/main.tex`: current conference source.
- `paper/manuscript/conference_final.pdf`: rebuilt supervisor review PDF.
- `paper/manuscript/main_full.tex` and `main_full.pdf`: plain-language full-results version (all twelve settings including the parallel design, every generated table plus the accuracy-change figure; regenerate with `python paper/manuscript/scripts/make_full_tables.py`).
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

## Setup (fresh clone)

```bash
git clone https://github.com/ihmorol/dt-nb-noise-removal.git
cd dt-nb-noise-removal
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt    # pandas<3 is load-bearing, see the file
java -version                                # Java 17+ (Weka jars are vendored in code/lib/)
```

## Repository layout

```
data/raw/                 the ten datasets, vendored (sources in data/README.md)
code/                     all experiment code (one script per experiment; see code/README.md)
  ├── e9_farid.py         E9f+E3b: the manuscript's Table II — 12 arms × 2 protocols, J48 world
  ├── e5_strategy.py      E5: graded noise handling (soft / correct / committee vs hard deletion)
  ├── e11_lr_mlp_hybrid.py E11: committee filter + Alg 2 on LR / PyTorch MLP (hybrid data)
  ├── replicate_farid.py  R1: faithful replication of Farid's Tables 8–11 (Weka J48 + FaithfulNB)
  ├── leakage_check.py    E3a: protocol A vs B vs C leakage audit (sklearn world)
  ├── main.py             E9p: the original sklearn pipeline pilot
  ├── faithful_nb.py      textbook NB (add-one nominal counts, Gaussian numeric)
  ├── weka_utils.py       ARFF writing, J48 runner, per-row prediction parser
  └── lib/                vendored Weka 3.8.6 jars (needs Java 17+)
results/                  every experiment's committed outputs (config.json, metrics, per-fold
                          rows, removal records, run logs)
paper/manuscript/         the audited conference paper (tables auto-generated from results/)
paper/draft-with-pilot-results/  the replication/leakage-audit draft
notes/                    experiment write-ups, project audit, literature ledger
trackers/                 experiment registry, todos, writing progress
CONTEXT.md                fixed domain vocabulary (the exact terms every file uses)
DECISIONS.md              append-only log: every design choice, pre-registered and dated
AGENTS.md                 how to work in this repo with an AI agent
```

## Data

The ten datasets of Farid et al. (2014), Table 5, are vendored under `data/raw/` (iris loads via
scikit-learn; NSL-KDD is `KDDTrain+_20Percent`). Sources, download dates, and the three known
deviations from the paper's Table 5 are documented in [`data/README.md`](data/README.md).

## Working on this with an AI agent

See [`AGENTS.md`](AGENTS.md): the read order, the pre-registration rule, and the integrity rules
that every table in the papers depends on.
