"""Run the pipeline on the datasets and write the results.

    data  ->  Path 1: Alg 1 -> Alg 2  ->  new data  ->  NB, DT
          ->  Path 2: Alg 2 -> Alg 1  ->  new data  ->  NB, DT

Every run writes, into ../results/EXP-E9_pipeline/:

    metrics.csv              accuracy and macro-F1 per dataset, arm and classifier
    stages.csv               what each algorithm removed, per arm, averaged over folds
    versions_removals.csv    what the whole-dataset pass removed
    versions/<dataset>/      the three data versions (main, Path 1, Path 2) + removals.txt

Usage:
    python main.py                       every dataset found under ../data/raw
    python main.py iris glass            a subset
    python main.py --seeds 3             fewer CV repeats
    python main.py --alpha 0.0 --nb mixed --support on
    python main.py --no-trace            skip the three-data-version dump
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from algorithm2 import CCP_ALPHA                                    # noqa: E402
from data import available, load_data                               # noqa: E402
from pipeline import N_SPLITS, describe_path, run_arm               # noqa: E402
from tables import (print_arm_header, print_arm_row,                # noqa: E402
                    print_farid_comparison, print_headline,
                    stage_rows_for, write_metrics, write_removals, write_stages)
from versions import trace_dataset                                  # noqa: E402

RESULTS = Path(__file__).resolve().parent.parent / "results" / "EXP-E9_pipeline"

# The arms that are run. "N" = Farid Algorithm 1, "A" = Farid Algorithm 2, in that order.
# The first six are references; the last four are the two paths with both classifiers.
ARMS = [
    ("baseline", (), "NB"), ("baseline", (), "DT"),
    ("Alg1", ("N",), "NB"), ("Alg1", ("N",), "DT"),
    ("Alg2", ("A",), "NB"), ("Alg2", ("A",), "DT"),
    ("C1 N->A", ("N", "A"), "NB"), ("C1 N->A", ("N", "A"), "DT"),
    ("C2 A->N", ("A", "N"), "NB"), ("C2 A->N", ("A", "N"), "DT"),
]

DEFAULTS = {
    "seeds": 5,
    "alpha": CCP_ALPHA,
    "support": True,
    "nb": "gaussian",
    "out": "metrics.csv",
    "trace": True,
}


def parse_args(argv):
    """Read `--name value` or `--name=value` options plus the dataset names."""
    options = dict(DEFAULTS)
    names = []
    i = 0
    while i < len(argv):
        argument = argv[i]

        if argument == "--no-trace":
            options["trace"] = False
            i += 1
        elif argument.startswith("--"):
            key = argument[2:]
            if "=" in key:
                key, value = key.split("=", 1)
                i += 1
            else:
                value = argv[i + 1]
                i += 2
            if key not in options:
                raise SystemExit(f"unknown option --{key}")
            options[key] = value
        else:
            names.append(argument)
            i += 1

    options["seeds"] = int(options["seeds"])
    options["alpha"] = float(options["alpha"])
    options["support"] = str(options["support"]).lower() in ("on", "true", "1")
    options["n_splits"] = N_SPLITS
    return names, options


def run_dataset(name, settings):
    """Cross-validate every arm on one dataset and return the results."""
    X, y, meta = load_data(name)
    print(f"\n--- {name}: {X.shape[0]} instances x {X.shape[1]} attributes, "
          f"{meta['n_classes']} classes ---")
    print_arm_header()

    results = {}
    for path, steps, final in ARMS:
        start = time.time()
        result = run_arm(X, y, steps, final, range(settings["seeds"]), settings["alpha"],
                         settings["support"], settings["nb"])
        results[f"{path}->{final}"] = result
        print_arm_row(path, final, result, time.time() - start)

    print("   what each algorithm removed (averaged over the folds):")
    print(f"     Path 1 (Alg 1 -> Alg 2): {describe_path(results['C1 N->A->DT']['stages'])}")
    print(f"     Path 2 (Alg 2 -> Alg 1): {describe_path(results['C2 A->N->DT']['stages'])}")
    return results


def main():
    names, settings = parse_args(sys.argv[1:])
    if not names:
        names = available()

    print("=" * 96)
    print("Farid (2014) pipeline -- data -> [Alg 1 -> Alg 2] and [Alg 2 -> Alg 1] -> NB / DT")
    print(f"{settings['n_splits']}-fold CV x {settings['seeds']} seeds, every stage refit "
          f"inside the training fold, tree ccp_alpha={settings['alpha']}, "
          f"NB likelihood={settings['nb']}, Alg-2 weights to NB={settings['support']}")
    print("=" * 96)

    all_results = {}
    stage_rows = []
    removal_rows = []

    for name in names:
        all_results[name] = run_dataset(name, settings)
        stage_rows += stage_rows_for(name, all_results[name])

        if settings["trace"]:
            report, rows = trace_dataset(name, RESULTS, settings["alpha"],
                                         settings["support"], settings["nb"])
            removal_rows += rows
            print("\n".join(report))

    print_headline(all_results)
    print_farid_comparison(all_results)

    RESULTS.mkdir(parents=True, exist_ok=True)
    write_metrics(RESULTS, settings["out"], all_results, settings)
    write_stages(RESULTS, stage_rows)
    print(f"\n[saved] {RESULTS / settings['out']}")
    print(f"[saved] {RESULTS / 'stages.csv'}")
    if removal_rows:
        write_removals(RESULTS, removal_rows)
        print(f"[saved] {RESULTS / 'versions_removals.csv'} and versions/<dataset>/")


if __name__ == "__main__":
    main()
