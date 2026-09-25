"""Load the ten datasets of Farid et al. (2014).

    X, y, meta = load_original("vote")   one column per attribute (the paper's view)
    X, y, meta = load_data("vote")       nominal attributes one-hot encoded

The raw files live in ../data/raw/.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import datasets as sklearn_datasets

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"

# How to read each dataset.
#   files    raw files, relative to data/raw (iris comes from sklearn instead)
#   label    position of the class column
#   drop     columns that are not attributes (row ids, difficulty scores)
#   nominal  the file stores nominal values as number codes
#   table5   (instances, attributes, classes) as printed in Farid's Table 5
DATASETS = {
    "breast-cancer": {"files": ["breast-cancer/breast-cancer.data"], "label": 0,
                      "table5": (286, 9, 2)},
    "contact-lenses": {"files": ["contact-lenses/lenses.data"], "label": -1, "drop": [0],
                       "sep": r"\s+", "nominal": True, "table5": (24, 4, 3)},
    "diabetes": {"files": ["diabetes.csv"], "label": -1, "table5": (768, 8, 2)},
    "glass": {"files": ["glass/glass.data"], "label": -1, "drop": [0],
              "table5": (214, 9, 7)},
    "iris": {"files": [], "table5": (150, 4, 3)},
    "soybean": {"files": ["soybean.csv"], "label": -1, "header": 0,
                "table5": (683, 35, 19)},
    "vote": {"files": ["vote/house-votes-84.data"], "label": 0, "table5": (435, 16, 2)},
    "image-segmentation": {"files": ["image-segmentation/segmentation.data",
                                     "image-segmentation/segmentation.test"],
                           "label": 0, "comment": ";", "skip": 4,
                           "table5": (1500, 19, 7)},
    "tic-tac-toe": {"files": ["tic-tac-toe/tic-tac-toe.data"], "label": -1,
                    "table5": (958, 9, 2)},
    "nsl-kdd": {"files": ["nsl-kdd-train20.txt"], "label": 41, "drop": [42],
                "table5": (25192, 41, 23)},
}

# Farid (2014) Tables 8-11: 10-fold CV accuracy (%) of C4.5 ("DT"), Algorithm 1,
# Naive Bayes and Algorithm 2.
FARID_2014 = {
    "breast-cancer": {"DT": 75.52, "Alg1": 81.46, "NB": 71.67, "Alg2": 75.87},
    "contact-lenses": {"DT": 83.33, "Alg1": 91.66, "NB": 70.83, "Alg2": 87.50},
    "diabetes": {"DT": 73.82, "Alg1": 79.03, "NB": 76.30, "Alg2": 85.41},
    "glass": {"DT": 66.82, "Alg1": 76.27, "NB": 48.59, "Alg2": 52.33},
    "iris": {"DT": 96.00, "Alg1": 98.66, "NB": 96.00, "Alg2": 98.00},
    "soybean": {"DT": 91.50, "Alg1": 92.97, "NB": 92.83, "Alg2": 94.15},
    "vote": {"DT": 96.32, "Alg1": 97.70, "NB": 90.11, "Alg2": 94.48},
    "image-segmentation": {"DT": 95.73, "Alg1": 96.53, "NB": 81.06, "Alg2": 85.19},
    "tic-tac-toe": {"DT": 85.07, "Alg1": 88.10, "NB": 69.62, "Alg2": 78.91},
    "nsl-kdd": {"DT": 71.11, "Alg1": 81.92, "NB": 76.27, "Alg2": 82.39},
}


def available():
    """Names of the datasets whose raw files are on disk."""
    return [name for name, d in DATASETS.items()
            if all((RAW / file).exists() for file in d["files"])]


def read_table(name):
    """Read the raw file(s) of one dataset. Returns (attribute table, class labels)."""
    if name == "iris":
        bunch = sklearn_datasets.load_iris(as_frame=True)
        table = bunch.data
        table.columns = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
        return table, bunch.target.astype(str).to_numpy()

    d = DATASETS[name]
    parts = [pd.read_csv(RAW / file, sep=d.get("sep", ","), header=d.get("header"),
                         comment=d.get("comment"), skiprows=d.get("skip", 0))
             for file in d["files"]]
    table = pd.concat(parts, ignore_index=True)

    labels = table.iloc[:, d["label"]].astype(str).str.strip().to_numpy()
    not_attributes = [d["label"]] + d.get("drop", [])
    table = table.drop(columns=table.columns[not_attributes])

    # some UCI files pad their text values with spaces
    for column in table.columns:
        if table[column].dtype == object:
            table[column] = table[column].str.strip()
    if d.get("nominal"):
        table = table.astype(str)
    return table, labels


def load_original(name):
    """One column per attribute, the way the paper and Weka see the data.

    A nominal column holds the position of its value in meta["levels"][column].
    A numeric column holds the number itself.
    meta["nominal"] says which columns are nominal.
    """
    table, y = read_table(name)
    stored_as_codes = DATASETS[name].get("nominal", False)

    columns, nominal, levels = [], [], []
    for column in table.columns:
        values = table[column]
        if values.dtype == object and not stored_as_codes:
            numbers = pd.to_numeric(values, errors="coerce")
            if numbers.notna().all():          # text that is really numbers
                values = numbers

        if values.dtype == object:
            categories = pd.Categorical(values.fillna("?"))
            columns.append(categories.codes.astype(float))
            nominal.append(True)
            levels.append([str(level) for level in categories.categories])
        else:
            columns.append(values.to_numpy(dtype=float))
            nominal.append(False)
            levels.append(None)

    X = np.column_stack(columns)
    meta = {
        "name": name,
        "names": [str(column) for column in table.columns],
        "nominal": np.array(nominal),
        "levels": levels,
        "n_classes": len(np.unique(y)),
        "table5": DATASETS[name]["table5"],
    }
    return X, y, meta


def load_data(name):
    """The same data with every nominal attribute turned into 0/1 columns (one-hot).

    The numeric columns come first, then the 0/1 columns, named "attribute_value".
    """
    X, y, meta = load_original(name)

    table = pd.DataFrame()
    for j, column in enumerate(meta["names"]):
        if meta["nominal"][j]:
            table[column] = pd.Categorical.from_codes(X[:, j].astype(int),
                                                      meta["levels"][j])
        else:
            table[column] = X[:, j]
    table = pd.get_dummies(table)

    meta["columns"] = [str(column) for column in table.columns]
    return table.to_numpy(dtype=float), y, meta


if __name__ == "__main__":
    for name in available():
        X, y, meta = load_original(name)
        shape = (X.shape[0], X.shape[1], meta["n_classes"])
        note = "" if shape == meta["table5"] else "   <- differs from Table 5"
        print(f"{name:20s} {shape[0]:6d} rows x {shape[1]:3d} attributes, "
              f"{shape[2]:3d} classes   Table 5: {meta['table5']}{note}")
