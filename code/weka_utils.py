"""Run Weka from Python: write ARFF files, run J48 (C4.5) and NaiveBayes, read the output.

J48 is Weka's C4.5. It splits a nominal attribute into one branch per value, so
"attribute" means one ARFF column, as in the paper's Algorithm 2.

J48 prints its tree like this; each "|   " is one level deeper:
    petal_width <= 0.6: Iris-setosa (50.0)
    petal_width > 0.6
    |   petal_width <= 1.7
"""
import os
import re
import subprocess
from pathlib import Path

LIB = Path(__file__).resolve().parent / "lib"
CLASSPATH = os.pathsep.join(str(jar) for jar in sorted(LIB.glob("*.jar")))
J48_OPTIONS = ["-C", "0.25", "-M", "2"]          # Weka's defaults


def clean_name(name):
    """Attribute names with only letters, digits and _ (safe for ARFF and tree parsing)."""
    return re.sub(r"[^A-Za-z0-9_]", "_", str(name))


def write_arff(path, X, y, meta, class_values, columns=None, weights=None):
    """Write the rows of X and y as an ARFF file. The class is the last attribute.

    class_values should be the classes of the WHOLE dataset, so every file of a
    dataset has the same header, even a training fold that misses a class.
    columns selects a subset of the attributes to write (default: all); X stays
    the full-width array and is indexed by the selected columns. weights gives
    one Weka instance weight per row, written as ARFF's trailing {weight}.
    """
    if columns is None:
        columns = range(len(meta["names"]))
    with open(path, "w", encoding="utf-8") as file:
        file.write("@relation data\n\n")
        for j in columns:
            name = meta["names"][j]
            if meta["nominal"][j]:
                values = ",".join("'" + value.replace("'", "''") + "'"
                                  for value in meta["levels"][j])
                file.write(f"@attribute {clean_name(name)} {{{values}}}\n")
            else:
                file.write(f"@attribute {clean_name(name)} real\n")
        file.write(f"@attribute class {{{','.join(class_values)}}}\n\n@data\n")

        for i, (row, label) in enumerate(zip(X, y)):
            cells = []
            for j in columns:
                value = row[j]
                if meta["nominal"][j]:
                    cells.append("'" + meta["levels"][j][int(value)].replace("'", "''") + "'")
                elif float(value).is_integer():
                    cells.append(str(int(value)))
                else:
                    cells.append(f"{value:.10g}")
            cells.append(f"'{label}'")
            line = ",".join(cells)
            if weights is not None:
                line += f" {{{weights[i]:.10g}}}"
            file.write(line + "\n")


def run_weka(arguments):
    command = ["java", "-cp", CLASSPATH] + arguments
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8",
                            errors="replace", timeout=600)
    if result.returncode != 0:
        raise RuntimeError(f"Weka failed: {' '.join(arguments[:2])}\n{result.stderr[-2000:]}")
    return result.stdout


def test_accuracy(output):
    """The 'Correctly Classified Instances' % of the last evaluation in the output."""
    found = re.findall(r"Correctly Classified Instances\s+\d+\s+([\d.]+)", output)
    if not found:
        raise RuntimeError("no accuracy line in the Weka output")
    return float(found[-1])


def run_j48(train_file, test_file, unpruned=False):
    """Train J48 on train_file, test on test_file.

    Returns (accuracy %, tree lines, confusion matrix). With a -T test file the
    output holds two evaluations (training and test); the accuracy regex and the
    confusion-matrix parser below both take the LAST one, which is the test set.
    """
    options = J48_OPTIONS + (["-U"] if unpruned else [])
    output = run_weka(["weka.classifiers.trees.J48"] + options
                      + ["-t", str(train_file), "-T", str(test_file)])

    start = output.find("J48 unpruned tree" if unpruned else "J48 pruned tree")
    end = output.find("Number of Leaves", start)
    if start < 0 or end < 0:
        raise RuntimeError("no tree in the J48 output")
    return test_accuracy(output), output[start:end].splitlines(), confusion_matrix(output)


def confusion_matrix(output):
    """The last confusion matrix in Weka's output, as counts[actual][predicted].

    Weka prints it as
        === Confusion Matrix ===
          a  b  c   <-- classified as
         50  0  0 |  a = setosa
    one row per class of the dataset header, so a class absent from the test
    fold still gets a row of zeros.
    """
    rows = []
    for line in output[output.rfind("=== Confusion Matrix ==="):].splitlines()[1:]:
        counts = line.split("|")[0].split()
        if counts and all(token.isdigit() for token in counts):
            rows.append([int(token) for token in counts])
        elif rows:
            break                       # the matrix ended ("Time taken...", etc.)
    k = len(rows)
    if k == 0 or any(len(row) != k for row in rows):
        raise RuntimeError("could not parse a square confusion matrix")
    return rows


def tree_depths(tree_lines):
    """{attribute name: smallest depth where the tree tests it}, root = depth 1."""
    depths = {}
    for line in tree_lines:
        text = line.replace("|   ", "").strip()
        if not any(test in text for test in (" <= ", " > ", " = ")):
            continue                        # header or separator line
        attribute = text.split()[0]
        depth = line.count("|   ") + 1
        depths[attribute] = min(depth, depths.get(attribute, depth))
    return depths


def weka_predicted(classifier, train_file, test_file, options=()):
    """The predicted class of every test row, from the -p 0 output (file order).

    Prediction lines look like:   3   1:no   2:yes   +   0.9
    """
    output = run_weka([classifier] + list(options) + ["-t", str(train_file),
                                                      "-T", str(test_file), "-p", "0"])
    predicted = []
    for line in output[output.find("inst#"):].splitlines()[1:]:
        labels = [token.split(":", 1)[1] for token in line.split() if ":" in token]
        if len(labels) >= 2:
            predicted.append(labels[1])
    return predicted


def naive_bayes_mistakes(train_file, test_file):
    """Run Weka's NaiveBayes. Returns one True/False per test row: True = wrong."""
    output = run_weka(["weka.classifiers.bayes.NaiveBayes", "-t", str(train_file),
                       "-T", str(test_file), "-p", "0"])
    # prediction lines look like:   3   1:no   2:yes   +   0.9
    wrong = []
    for line in output[output.find("inst#"):].splitlines()[1:]:
        labels = [token.split(":", 1)[1] for token in line.split() if ":" in token]
        if len(labels) >= 2:
            wrong.append(labels[0] != labels[1])
    return wrong
