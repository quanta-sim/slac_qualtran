"""Run the portable SLAC benchmarks without a notebook."""

import argparse
from pathlib import Path

import matplotlib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--max-n", type=int, default=16)
    args = parser.parse_args()
    if args.max_n < 3:
        parser.error("--max-n must be at least 3")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    from .analysis import (
        benchmark_complete_inverse_square_condition,
        compare_three_slac_circuits,
        validate_complete_inverse_square_action,
    )
    from .circuits import CompleteMultiscaleInverseSquare, inverse_square_column
    from .resources import slac_rotation_audit, theoretical_component_table

    args.output.mkdir(parents=True, exist_ok=True)
    comparison, figure = compare_three_slac_circuits(range(3, args.max_n + 1))
    comparison.to_csv(args.output / "toffoli_comparison.csv", index=False)
    figure.savefig(args.output / "toffoli_comparison.png", dpi=180)
    figure.savefig(args.output / "toffoli_comparison.pdf")
    plt.close(figure)
    checks = [validate_complete_inverse_square_action(n, 3) for n in (3, 4)]
    pd.DataFrame([{k: v for k, v in check.items() if not isinstance(v, np.ndarray)}
                  for check in checks]).to_csv(args.output / "action_checks.csv", index=False)
    benchmark_complete_inverse_square_condition(range(3, min(args.max_n, 7) + 1)).to_csv(
        args.output / "conditioning.csv", index=False)
    rotations = pd.DataFrame([
        {"n": n, **slac_rotation_audit(CompleteMultiscaleInverseSquare(n, 2*n + 4, True))}
        for n in range(3, args.max_n + 1)
    ]).fillna(0)
    rotations.to_csv(args.output / "rotations.csv", index=False)
    theoretical_component_table(args.max_n, 2*args.max_n + 4).to_csv(
        args.output / "components.csv", index=False)
    precision = []
    for r in range(2, 17):
        error = float(np.max(np.abs(np.fft.fft(inverse_square_column(4, r) - inverse_square_column(4)))))
        precision.append({"n_ref": r, "operator_error": error, "bound": 7 / (1 << r)})
    pd.DataFrame(precision).to_csv(args.output / "precision.csv", index=False)
    print(f"Saved results to {args.output}")


if __name__ == "__main__":
    main()
