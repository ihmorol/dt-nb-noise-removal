# Datasets — the seed paper's Table 5 benchmark

Target shapes (verified from the PDF, Table 5 p. 1942):

| Dataset | Attr | Types | Instances | Classes | Source |
|---|---|---|---|---|---|
| breast-cancer | 9 | nominal | 286 | 2 | UCI "Breast Cancer" (Wisconsin 1988 original — **NOT** the 699-row WBCD) |
| contact-lenses | 4 | nominal | 24 | 3 | UCI "Lens" |
| diabetes | 8 | real | 768 | 2 | UCI "Pima Indians Diabetes" |
| glass | 9 | real | 214 | 7 | UCI "Glass Identification" (drop Id column) |
| iris | 4 | real | 150 | 3 | UCI/sklearn built-in |
| soybean | 35 | nominal | 683 | 19 | UCI "Soybean (Large)" — train(307)+test(376) combined |
| vote | 16 | nominal | 435 | 2 | UCI "Congressional Voting Records" (drop the 0th name column) |
| image-segmentation | 19 | real | 1500 | 7 | UCI "Image Segmentation" |
| tic-tac-toe | 9 | nominal | 958 | 2 | UCI "Tic-Tac-Toe Endgame" |
| nsl-kdd | 41 | mixed | 25192 | 23 | UNB/CIC NSL-KDD — looks like `KDDTrain+_20Percent` (25,192 rows); classify the full 23 labels, not binary |

## Rules

- Keep raw files in `data/raw/<name>/` untouched; loaders do cleaning.
- Record each file's URL + download date + row/col counts here after E0 passes.
- Missing-value policy and numeric-vs-nominal handling must be identical across all models in one experiment (part of the config, not the loader's choice).

## What is on disk (downloaded 2026-09-18)

| Local path | Source | Raw rows | After `load_data` |
|---|---|---|---|
| `raw/breast-cancer/breast-cancer.data` | archive.ics.uci.edu/static/public/14 | 286 | 286 × 41 (9 nominal → one-hot) |
| `raw/contact-lenses/lenses.data` | copied from `notes/pilots/data/` | 24 | 24 × 4 |
| `raw/diabetes.csv` | raw.githubusercontent.com/jbrownlee/Datasets (Pima, headerless) | 768 | 768 × 8 |
| `raw/glass/glass.data` | archive.ics.uci.edu/static/public/42 | 214 | 214 × 9 (Id dropped) |
| `raw/iris/` (sklearn) | `sklearn.datasets.load_iris` | 150 | 150 × 4 |
| `raw/soybean.csv` | openml.org dataset 42 (soybean, 683 × 35) | 683 | 683 × 133 (35 nominal → one-hot) |
| `raw/vote/house-votes-84.data` | archive.ics.uci.edu/static/public/105 | 435 | 435 × 48 (16 nominal, '?' = own category) |
| `raw/image-segmentation/segmentation.{data,test}` | archive.ics.uci.edu/static/public/50 | 210 + 2100 | 2310 × 19 |
| `raw/tic-tac-toe/tic-tac-toe.data` | copied from `notes/pilots/data/` | 958 | 958 × 27 (9 nominal → one-hot) |
| `raw/nsl-kdd-train20.txt` | github.com/defcom17/NSL_KDD (`KDDTrain+_20Percent.txt`) | 25192 | 25192 × 118 (3 nominal → one-hot, difficulty dropped) |

Known shape differences from Table 5, all resolved the same way by the loader (run
`python code/data.py` to print them): nominal attributes are one-hot encoded, so the
attribute count grows on nominal datasets; UCI glass has 6 classes (Table 5 says 7);
`KDDTrain+_20Percent` contains 22 of the 23 label types; image-segmentation is the full
2310-row UCI set (Table 5 says 1500 — the paper's subset is unknown).
