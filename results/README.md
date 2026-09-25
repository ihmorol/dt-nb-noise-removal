# Results artifacts

One folder per experiment: `EXP-<id>_<shortname>/`

```
results/
└── EXP-E3a_leakage-audit/
    ├── config.json          # full config incl. seeds — copied by the run script
    ├── metrics.csv          # one row per (dataset, model, protocol, seed)
    ├── summary.md           # 5 lines: the answer to the experiment's ONE question
    └── figures/             # png only, 300dpi, also used in the paper
```

Rule: if `summary.md` can't answer the experiment's question in ≤5 lines, the experiment wasn't designed tightly enough — rerun it narrower.
