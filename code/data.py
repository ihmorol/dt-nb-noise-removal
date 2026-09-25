"""Data selection and structuring.

Everything in the project gets its data through one function:

    X, y, meta = load_data("tic-tac-toe")

X    instances x attributes, float ndarray (nominal attributes one-hot encoded)
y    class labels, ndarray of strings
meta dict with the loaded shape and the shape Farid (2014) Table 5 reports

Raw files live in ../data/raw/<dataset>/. Nothing else in the code touches files.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parent.parent / "data" / "raw"


@dataclass(frozen=True)
class Dataset:
    name: str
    files: tuple[str, ...] = ()      # relative to data/raw
    class_col: int = -1              # position of the label column in the file
    drop_cols: tuple[int, ...] = ()  # e.g. row ids, difficulty scores
    sep: str = ","
    header: int | None = None
    comment: str | None = None
    skiprows: int = 0                # e.g. a header line that lists only the attributes
    sklearn: str | None = None       # bundled datasets load from sklearn instead
    table5: tuple[int, int, int] = ()  # Farid Table 5: (instances, attributes, classes)
    force_nominal: bool = False      # file stores numeric CODES for nominal attributes


DATASETS = {
    "breast-cancer": Dataset(
        "breast-cancer", ("breast-cancer/breast-cancer.data",), class_col=0,
        table5=(286, 9, 2)),
    "contact-lenses": Dataset(
        "contact-lenses", ("contact-lenses/lenses.data",), class_col=-1, drop_cols=(0,),
        sep=r"\s+", table5=(24, 4, 3), force_nominal=True),
    "diabetes": Dataset(
        "diabetes", ("diabetes.csv",), class_col=-1, table5=(768, 8, 2)),
    "glass": Dataset(
        "glass", ("glass/glass.data",), class_col=-1, drop_cols=(0,), table5=(214, 9, 7)),
    "iris": Dataset(
        "iris", sklearn="iris", table5=(150, 4, 3)),
    "soybean": Dataset(
        "soybean", ("soybean.csv",), class_col=-1, header=0, table5=(683, 35, 19)),
    "vote": Dataset(
        "vote", ("vote/house-votes-84.data",), class_col=0, table5=(435, 16, 2)),
    "image-segmentation": Dataset(
        "image-segmentation",
        ("image-segmentation/segmentation.data", "image-segmentation/segmentation.test"),
        class_col=0, comment=";", skiprows=4, table5=(1500, 19, 7)),
    "tic-tac-toe": Dataset(
        "tic-tac-toe", ("tic-tac-toe/tic-tac-toe.data",), class_col=-1, table5=(958, 9, 2)),
    "nsl-kdd": Dataset(
        "nsl-kdd", ("nsl-kdd-train20.txt",), class_col=41, drop_cols=(42,),  # 42 = difficulty
        table5=(25192, 41, 23)),
}

# Farid Tables 8-11, 10-fold CV accuracy (%): C4.5, Algorithm 1, NB, Algorithm 2.
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


def _read_files(files, separator, header, comment, skiprows):
    """Read one or more raw files of a dataset into a single table."""
    engine = "python" if separator != "," else "c"       # the whitespace one needs python
    parts = []
    for file in files:
        part = pd.read_csv(RAW / file, sep=separator, header=header, comment=comment,
                           skiprows=skiprows, engine=engine)
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def _strip_spaces(table):
    """Some UCI files pad their text values; remove the padding."""
    for column in table.columns:
        if table[column].dtype == object:
            table[column] = table[column].str.strip()
    return table


def available() -> list[str]:
    """Dataset names whose raw files are on disk (sklearn ones always are)."""
    out = []
    for name, d in DATASETS.items():
        if d.sklearn is not None or all((RAW / f).exists() for f in d.files):
            out.append(name)
    return out


def load_data(name: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load one dataset, select its attributes, structure it for the pipeline."""
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}; available: {available()}")
    d = DATASETS[name]

    if d.sklearn is not None:
        from sklearn import datasets as sk
        bunch = getattr(sk, f"load_{d.sklearn}")(as_frame=True)
        frame, y = bunch.data, bunch.target
    else:
        missing = [f for f in d.files if not (RAW / f).exists()]
        if missing:
            raise FileNotFoundError(f"{name}: missing {missing} under {RAW}")
        frame = _read_files(d.files, d.sep, d.header, d.comment, d.skiprows)

        # the label column, then the features (minus any id or extra columns)
        y = frame.iloc[:, d.class_col].astype(str)
        label_column = frame.columns[d.class_col]
        extra_columns = [frame.columns[i] for i in d.drop_cols]
        frame = frame.drop(columns=[label_column] + extra_columns)

    frame = _strip_spaces(frame)
    y = pd.Series(y).map(lambda value: str(value).strip()).to_numpy()

    # nominal attributes are one-hot encoded, so every model sees numbers
    has_text = len(frame.select_dtypes(include="object").columns) > 0
    if has_text:
        frame = pd.get_dummies(frame)
    X = frame.to_numpy(dtype=float)

    meta = {
        "name": name,
        "n_instances": X.shape[0],
        "n_attributes": X.shape[1],
        "n_classes": int(np.unique(y).size),
        "columns": [str(c) for c in frame.columns],   # attribute names, one per X column
        "table5": d.table5,
    }
    return X, y, meta


def load_original(name: str) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load one dataset in its ORIGINAL attribute space -- no one-hot encoding.

    This is the representation the paper works in: one column per attribute, nominal
    attributes kept whole (so J48 can split one branch per value and Algorithm 2's
    depth is a depth over the paper's attributes, not over dummy columns).

    Returns:
        X             float ndarray; nominal columns hold integer level codes,
                      numeric columns their value
        y             class labels (strings)
        meta          the load_data meta dict plus:
                          names        attribute name per column
                          nominal      bool per column
                          levels       per nominal column, the level strings
                                      (taken from the WHOLE file -- the same
                                      information an ARFF header declares)
    """
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}; available: {available()}")
    d = DATASETS[name]

    if d.sklearn is not None:
        from sklearn import datasets as sk
        bunch = getattr(sk, f"load_{d.sklearn}")(as_frame=True)
        frame, y = bunch.data.astype(float), bunch.target
        # sklearn's column names contain spaces and brackets; give ARFF clean ones
        frame.columns = [c.split("(")[0].strip().replace(" ", "_")
                         for c in frame.columns]
    else:
        missing = [f for f in d.files if not (RAW / f).exists()]
        if missing:
            raise FileNotFoundError(f"{name}: missing {missing} under {RAW}")
        frame = _read_files(d.files, d.sep, d.header, d.comment, d.skiprows)
        y = frame.iloc[:, d.class_col].astype(str)
        frame = frame.drop(columns=[frame.columns[d.class_col]]
                                 + [frame.columns[i] for i in d.drop_cols])

    frame = _strip_spaces(frame)
    y = pd.Series(y).map(lambda value: str(value).strip()).to_numpy()

    # a file that stores nominal attributes as numeric codes (Table 5 declares
    # them nominal) is put back into nominal form before the type detection
    if d.force_nominal:
        frame = frame.astype(str)

    names, columns, nominal_mask, levels = [], [], [], []
    for column in frame.columns:
        series = frame[column]
        if series.dtype == object and not d.force_nominal:  # numeric-looking text?
            as_number = pd.to_numeric(series, errors="coerce")
            if as_number.notna().all():
                series = as_number
        if series.dtype == object:                      # still text -> nominal
            series = series.fillna("?")
            cats = pd.Categorical(series)
            names.append(str(column))
            nominal_mask.append(True)
            levels.append([str(v) for v in cats.categories])
            columns.append(cats.codes.astype(float))
        else:
            names.append(str(column))
            nominal_mask.append(False)
            levels.append(None)
            columns.append(series.to_numpy(dtype=float))

    X = np.column_stack(columns) if columns else np.empty((len(y), 0))
    meta = {
        "name": name,
        "n_instances": X.shape[0],
        "n_attributes": X.shape[1],
        "n_classes": int(np.unique(y).size),
        "columns": names,
        "table5": d.table5,
        "names": names,
        "nominal": np.asarray(nominal_mask, dtype=bool),
        "levels": levels,
    }
    return X, y, meta


if __name__ == "__main__":
    for name in available():
        X, y, meta = load_data(name)
        flag = "" if meta["table5"] == (X.shape[0], X.shape[1], meta["n_classes"]) else "  <- differs from Table 5"
        print(f"{name:20s} {X.shape[0]:6d} x {X.shape[1]:4d}  classes={meta['n_classes']:3d}"
              f"   Table5={meta['table5']}{flag}")
