"""Weka plumbing: write ARFF, run Weka's J48 and NaiveBayes from the command line,
and read back the two things the replication needs -- the accuracy and the tree
structure (which attributes the tree tests, at what minimum depth).

Why J48: the paper's Algorithm 1/2 describe a "C4.5-style" tree built by their own
Java code. J48 IS C4.5 (Weka's implementation), so it is the closest defensible
engine -- much closer than sklearn's CART (binary splits, no gain ratio). J48 also
splits nominal attributes one branch per value, which is what makes the paper's
"attribute" notion (one attribute = one ARFF column) the natural unit for the
depth weights, exactly as Algorithm 2 assumes.

Tree output format parsed here (example):
    petalwidth <= 0.6: Iris-setosa (50.0)
    petalwidth > 0.6
    |   petalwidth <= 1.7
    ...
The number of '|   ' prefixes is the node depth - 1, and the first token of the
test is the attribute name (names are sanitized to contain no spaces).
"""

import os
import re
import subprocess
from pathlib import Path

LIB = Path(__file__).resolve().parent / "lib"
# every jar in lib/ (weka + its deps such as bounce), joined the way this OS wants
WEKA_CLASSPATH = os.pathsep.join(str(jar) for jar in sorted(LIB.glob("*.jar")))
J48_OPTIONS = ["-C", "0.25", "-M", "2"]      # Weka's own defaults, fixed a priori


def sanitize(name):
    """Make an attribute name safe for ARFF and for tree-output parsing."""
    return re.sub(r"[^A-Za-z0-9_]", "_", str(name))


def write_arff(path, X, y, names, nominal_mask, levels, relation="data",
               class_levels=None):
    """Write one ARFF file. Class goes last, as Weka expects by default.

    X  float ndarray; nominal columns hold integer level codes
    levels  per nominal column: the level strings (must cover train AND test,
            which holds because they come from the whole file, exactly like an
            ARFF header declares all values up front)
    class_levels  the class values to declare. Pass the WHOLE-dataset levels so
            every file of a dataset shares one header -- a fold whose training
            slice misses a class still declares it, and Weka then (correctly)
            counts its test rows as errors instead of crashing.
    """
    names = [sanitize(n) for n in names]
    if class_levels is None:
        class_levels = sorted(set(str(v) for v in y))
    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write(f"@relation {sanitize(relation)}\n\n")
        for j, name in enumerate(names):
            if nominal_mask[j]:
                values = ",".join(f"'{str(v).replace(chr(39), chr(39) * 2)}'"
                                  for v in levels[j])
                f.write(f"@attribute {name} {{{values}}}\n")
            else:
                f.write(f"@attribute {name} real\n")
        f.write(f"@attribute class {{{','.join(class_levels)}}}\n\n")
        f.write("@data\n")
        for i in range(len(X)):
            row = []
            for j in range(X.shape[1]):
                if nominal_mask[j]:
                    row.append(f"'{levels[j][int(X[i, j])]}'")
                else:
                    value = X[i, j]
                    row.append(str(int(value)) if float(value).is_integer()
                               else f"{value:.10g}")
            row.append(f"'{y[i]}'")
            f.write(",".join(row) + "\n")


def _run(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                            errors="replace", timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"weka failed ({' '.join(cmd[:4])}...):\n{result.stderr[-2000:]}")
    return result.stdout


def _accuracy(output):
    """The 'Correctly Classified Instances' percentage from a Weka summary."""
    matches = re.findall(r"Correctly Classified Instances\s+\d+\s+([\d.]+)", output)
    if not matches:
        raise RuntimeError("no accuracy line in Weka output")
    return float(matches[-1])            # last = the evaluation section we asked for


def run_j48(train_arff, test_arff, unpruned=False):
    """Fit J48 on train, evaluate on test. Returns (accuracy, tree_lines)."""
    cmd = ["java", "-cp", WEKA_CLASSPATH, "weka.classifiers.trees.J48"] + J48_OPTIONS
    if unpruned:
        cmd.append("-U")
    cmd += ["-t", str(train_arff), "-T", str(test_arff)]
    output = _run(cmd)

    start = output.find("J48 pruned tree")
    if start < 0:
        start = output.find("J48 unpruned tree")
    if start < 0:
        raise RuntimeError("no tree section in J48 output")
    end = output.find("Number of Leaves", start)
    tree_lines = output[start:end].splitlines()
    return _accuracy(output), tree_lines


def parse_tree(tree_lines):
    """Tested attributes and their minimum depth (root = 1), per Algorithm 2.

    Returns {attribute_name: min_depth}. Lines that carry no test (headers,
    the size summary) are skipped.
    """
    tested = {}
    for line in tree_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith(("J48", "-----")):
            continue
        depth = line.count("|   ") + 1
        text = stripped.replace("|   ", "").strip()
        tokens = text.split()
        if not tokens:
            continue
        # a test line looks like 'attr <= 5: class (n)' or 'attr = value'; leaf
        # lines keep the test, the size summary lines ('Number of Leaves') do not
        # appear inside the tree body, so any token before an operator is the attr
        if not any(op in text for op in ("<=", " = ", " > ")):
            continue
        attribute = tokens[0]
        if attribute not in tested or depth < tested[attribute]:
            tested[attribute] = depth
    return tested


def run_naive_bayes(train_arff, test_arff, predictions=False):
    """Run Weka's NB. Returns (accuracy, wrong_flags).

    wrong_flags (only with predictions=True) is one bool per test row in file
    order -- True where Weka's prediction differs from the recorded label. This
    is what Algorithm 1's judge needs on the training data.
    """
    cmd = ["java", "-cp", WEKA_CLASSPATH, "weka.classifiers.bayes.NaiveBayes",
           "-t", str(train_arff), "-T", str(test_arff)]
    if predictions:
        cmd += ["-p", "0"]
    output = _run(cmd)

    if not predictions:
        return _accuracy(output), None

    # with -p, Weka prints ONLY the prediction table (no summary), so the
    # accuracy comes from the flags themselves
    header = output.find("inst#")
    wrong = []
    for line in output[header:].splitlines()[1:]:
        tokens = line.split()
        labelled = [t for t in tokens if ":" in t and not t.startswith("inst#")]
        if not tokens or len(labelled) < 2:
            continue
        actual = labelled[0].split(":", 1)[1]
        predicted = labelled[1].split(":", 1)[1]
        wrong.append(actual != predicted)
    accuracy = 100.0 * (1.0 - float(np_like_mean(wrong)))
    return accuracy, wrong


def np_like_mean(flags):
    """Mean of a list of bools without importing numpy here."""
    return sum(1 for f in flags if f) / len(flags) if flags else 0.0
