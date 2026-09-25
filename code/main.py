"""Run the pipeline on the datasets and save the results.

    data -> Path 1: Alg 1 -> Alg 2 -> NB, DT
         -> Path 2: Alg 2 -> Alg 1 -> NB, DT

Output, in ../results/EXP-E9_pipeline/:
    metrics.csv              accuracy and macro-F1 per dataset, arm and classifier
    stages.csv               what each algorithm removed, averaged over the folds
    versions_removals.csv    what the one pass over the whole dataset removed
    versions/<dataset>/      the data before and after each path, plus removals.txt

Examples:
    python main.py                      all datasets, the final settings
    python main.py iris glass --seeds 3
    python main.py --alpha 0.01 --nb gaussian --out metrics_alpha0.01.csv
"""
import argparse
import time
from pathlib import Path

from data import available, load_data
from pipeline import N_SPLITS, describe_path, run_arm
from tables import (METRICS_HEADER, REMOVALS_HEADER, STAGES_HEADER, metrics_rows,
                    print_arm_header, print_arm_row, print_farid_comparison,
                    print_headline, stage_rows, write_csv)
from versions import trace_dataset

RESULTS = Path(__file__).resolve().parent.parent / "results" / "EXP-E9_pipeline"

# (arm name, steps, final classifier). "N" = Algorithm 1, "A" = Algorithm 2.
# The first six are references, the last four are the two paths.
ARMS = [
    ("baseline", (), "NB"), ("baseline", (), "DT"),
    ("Alg1", ("N",), "NB"), ("Alg1", ("N",), "DT"),
    ("Alg2", ("A",), "NB"), ("Alg2", ("A",), "DT"),
    ("C1 N->A", ("N", "A"), "NB"), ("C1 N->A", ("N", "A"), "DT"),
    ("C2 A->N", ("A", "N"), "NB"), ("C2 A->N", ("A", "N"), "DT"),
]


def read_settings():
    parser = argparse.ArgumentParser(description="Farid (2014) two-path pipeline")
    parser.add_argument("datasets", nargs="*", help="default: every dataset on disk")
    parser.add_argument("--seeds", type=int, default=5, help="CV repeats (default 5)")
    parser.add_argument("--alpha", type=float, default=0.0,
                        help="Algorithm 2 tree pruning, sklearn ccp_alpha (default 0.0)")
    parser.add_argument("--nb", choices=["gaussian", "mixed"], default="mixed",
                        help="NB likelihood (default mixed)")
    parser.add_argument("--support", choices=["on", "off"], default="on",
                        help="pass Algorithm 2's weights to the judge and the final NB")
    parser.add_argument("--out", default="metrics.csv", help="name of the metrics file")
    parser.add_argument("--no-trace", action="store_true",
                        help="skip writing the data versions")
    settings = parser.parse_args()
    settings.n_splits = N_SPLITS
    return settings


def run_dataset(name, settings):
    """Cross-validate every arm on one dataset."""
    X, y, meta = load_data(name)
    print(f"\n--- {name}: {X.shape[0]} instances x {X.shape[1]} attributes, "
          f"{meta['n_classes']} classes ---")
    print_arm_header()

    results = {}
    for arm, steps, final in ARMS:
        start = time.time()
        result = run_arm(X, y, steps, final, range(settings.seeds), settings.alpha,
                         settings.support == "on", settings.nb)
        results[f"{arm}->{final}"] = result
        print_arm_row(arm, final, result, time.time() - start)

    print("   what each algorithm removed (averaged over the folds):")
    print(f"     Path 1: {describe_path(results['C1 N->A->DT']['stages'])}")
    print(f"     Path 2: {describe_path(results['C2 A->N->DT']['stages'])}")
    return results


def main():
    settings = read_settings()
    names = settings.datasets or available()

    print("=" * 96)
    print("Farid (2014) pipeline: [Alg 1 -> Alg 2] and [Alg 2 -> Alg 1], then NB or DT")
    print(f"{N_SPLITS}-fold CV x {settings.seeds} seeds, every step refit inside the "
          f"training fold, ccp_alpha={settings.alpha}, NB={settings.nb}, "
          f"weights to NB={settings.support}")
    print("=" * 96)

    all_results, all_stages, all_removals = {}, [], []
    for name in names:
        all_results[name] = run_dataset(name, settings)
        all_stages += stage_rows(name, all_results[name])
        if not settings.no_trace:
            report, removals = trace_dataset(name, RESULTS, settings.alpha,
                                             settings.support == "on", settings.nb)
            all_removals += removals
            print("\n".join(report))

    print_headline(all_results)
    print_farid_comparison(all_results)

    RESULTS.mkdir(parents=True, exist_ok=True)
    write_csv(RESULTS / settings.out, METRICS_HEADER, metrics_rows(all_results, settings))
    write_csv(RESULTS / "stages.csv", STAGES_HEADER, all_stages)
    print(f"\n[saved] {RESULTS / settings.out}")
    print(f"[saved] {RESULTS / 'stages.csv'}")
    if all_removals:
        write_csv(RESULTS / "versions_removals.csv", REMOVALS_HEADER, all_removals)
        print(f"[saved] {RESULTS / 'versions_removals.csv'} and versions/<dataset>/")


if __name__ == "__main__":
    main()
