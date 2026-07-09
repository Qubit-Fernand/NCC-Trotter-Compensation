#!/usr/bin/env python3
"""Compute step counts from the expected-precision constraint only.

This script separates the first HNCC resource-estimate constraint,

    r * per_step_error(T / r) <= epsilon,

from the sampling-overhead / 1-norm constraint.  It reuses the current
``hncc_resource_estimate.py`` implementation for Pauli decompositions and
per-step error curves, then writes one JSON file per system size.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Mapping, Sequence

import numpy as np

CORE_DIR = Path(__file__).resolve().parent / "cluster"
if str(CORE_DIR) not in sys.path:
    sys.path.insert(0, str(CORE_DIR))

import hncc_resource_estimate as h


def parse_int_map(items: Sequence[str], defaults: Mapping[int, int]) -> dict[int, int]:
    out = dict(defaults)
    for item in items:
        key, value = item.split(":", 1)
        out[int(key)] = int(value)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--N", dest="n_values", nargs="+", type=int, default=[8, 10], help="Even system sizes.")
    parser.add_argument("--orders", nargs="+", type=int, default=[2, 4, 6], help="Plain Suzuki orders.")
    parser.add_argument("--ncc-orders", nargs="+", type=int, default=[2], help="Base Suzuki orders for unitary-level NCC.")
    parser.add_argument("--hncc-orders", nargs="+", type=int, default=[2], help="Base Suzuki orders to compensate.")
    parser.add_argument("--q0", nargs="*", default=[], help="Override q0 by order, e.g. --q0 2:6 4:10.")
    parser.add_argument("--s0", nargs="*", default=[], help="Override s0 by order, e.g. --s0 2:6 4:10.")
    parser.add_argument("--extrap-bases", nargs="+", type=int, default=[2, 4], help="Base Suzuki orders for extrapolation.")
    parser.add_argument("--extrap-points", nargs="+", type=int, default=[3], help="Kept for clarity; only 3-point extrapolation is used.")
    parser.add_argument("--time", type=float, default=1.0, help="Simulation time for eps scan.")
    parser.add_argument("--eps", nargs="+", type=float, default=[1e-3, 8e-4, 6e-4, 4e-4, 3e-4, 2e-4, 1e-4], help="Target precisions.")
    parser.add_argument("--time-values", nargs="+", type=float, default=[0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0], help="Times for time scan.")
    parser.add_argument("--fixed-eps", type=float, default=1e-3, help="Precision for time scan.")
    parser.add_argument("--x-min", type=float, default=1e-3, help="Minimum step size in interpolation grid.")
    parser.add_argument("--x-max", type=float, default=1.0, help="Maximum step size in interpolation grid.")
    parser.add_argument("--num-x", type=int, default=40, help="Number of interpolation-grid points.")
    parser.add_argument("--scan", choices=("eps", "time", "both"), default="both", help="Which scans to collect.")
    parser.add_argument("--output-dir", type=Path, default=Path("resource/data/split_constraints"), help="Output directory.")
    parser.add_argument(
        "--seed-final",
        type=Path,
        default=None,
        help="Seed precision rows from an existing final resource JSON instead of recomputing step-error curves.",
    )
    return parser.parse_args()


def find_min_curve_steps(
    x_grid: np.ndarray,
    step_errors: np.ndarray,
    eps: float,
    time: float,
    method: str,
) -> tuple[int, float, float]:
    def feasible(steps: int) -> bool:
        x = time / steps
        return steps * h.log_interpolate_error(x_grid, step_errors, x) <= eps

    hi = 1
    while not feasible(hi):
        hi *= 2
        if hi > 10**8:
            raise RuntimeError(f"could not find precision step count for {method}, eps={eps:g}")
    lo = 0
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if feasible(mid):
            hi = mid
        else:
            lo = mid
    x = time / hi
    bias = hi * h.log_interpolate_error(x_grid, step_errors, x)
    return hi, x, bias


def precision_rows_for_time(
    result: h.EstimateResult,
    eps_values: Sequence[float],
    time: float,
    formula_orders: Sequence[int],
    extrap_formulas: Sequence[h.ExtrapolationFormula],
    scan_value: float | None = None,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for eps in eps_values:
        row_scan_value = eps if scan_value is None else scan_value
        for order in formula_orders:
            steps, x, bias = h.find_min_formula_steps(result, order, eps, time)
            rows.append(row("S" + str(order), row_scan_value, eps, time, steps, x, bias))

        for formula in extrap_formulas:
            steps, x, bias = h.find_min_extrapolation_steps(result, formula, eps, time)
            rows.append(row(formula.name, row_scan_value, eps, time, steps, x, bias))

        for comp in result.ncc.values():
            steps, x, bias = find_min_curve_steps(result.x_grid, comp.step_errors, eps, time, comp.method)
            rows.append(row(comp.method, row_scan_value, eps, time, steps, x, bias))

        for comp in result.hncc.values():
            steps, x, bias = find_min_curve_steps(result.x_grid, comp.step_errors, eps, time, comp.method)
            rows.append(row(comp.method, row_scan_value, eps, time, steps, x, bias))
    return rows


def row(method: str, scan_value: float, eps: float, time: float, steps: int, x: float, bias: float) -> dict[str, object]:
    return {
        "method": method,
        "scan_value": float(scan_value),
        "eps": float(eps),
        "time": float(time),
        "r_precision": int(steps),
        "x_precision": float(x),
        "bias_precision": float(bias),
    }


def output_path(output_dir: Path, n: int) -> Path:
    return output_dir / f"r_for_expected_precision_N{n}.json"


def seeded_precision_payload(path: Path, n: int) -> dict[str, object]:
    source = json.loads(path.read_text(encoding="utf-8"))
    metadata = dict(source.get("metadata", {}))
    metadata["constraint"] = "expected_precision"
    metadata["seed_final"] = str(path)

    def convert(row_data: Mapping[str, object], eps: float, time: float) -> dict[str, object]:
        return row(
            str(row_data["method"]),
            float(row_data["scan_value"]),
            eps,
            time,
            int(row_data["steps"]),
            float(row_data["x"]),
            float(row_data["bias"]),
        )

    eps_scan = {}
    if source.get("eps_scan"):
        eps_scan = {
            str(key): [convert(row_data, float(row_data["scan_value"]), float(metadata["time"])) for row_data in rows]
            for key, rows in source["eps_scan"].items()
        }

    time_scan = {}
    if source.get("time_scan"):
        fixed_eps = float(metadata.get("fixed_eps", 1e-3))
        for time_key, rows in source["time_scan"].items():
            converted = []
            for row_data in rows:
                converted.append(
                    convert(row_data, fixed_eps, float(time_key))
                )
            time_scan[str(time_key)] = converted

    metadata["n_values"] = [n]
    return {"metadata": metadata, "eps_scan": eps_scan, "time_scan": time_scan}


def main() -> None:
    args = parse_args()
    if any(n % 2 for n in args.n_values):
        raise ValueError("Use even N for the periodic even-odd decomposition.")
    if args.extrap_points != [3]:
        raise ValueError("This simplified comparison keeps only the 3-point extrapolation formula.")

    default_q0 = {order: 2 * order + 2 for order in args.hncc_orders}
    default_s0 = {order: 2 * order + 2 for order in args.hncc_orders}
    q0_by_order = parse_int_map(args.q0, default_q0)
    s0_by_order = parse_int_map(args.s0, default_s0)
    x_grid = np.logspace(math.log10(args.x_min), math.log10(args.x_max), args.num_x)
    extrap_formulas = [h.richardson_formula(base, (1, 2, 6)) for base in args.extrap_bases]
    formula_orders = sorted(set(args.orders) | set(args.extrap_bases) | set(args.ncc_orders) | set(args.hncc_orders))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.seed_final is not None:
        if len(args.n_values) != 1:
            raise ValueError("--seed-final expects exactly one --N value.")
        n = args.n_values[0]
        path = output_path(args.output_dir, n)
        path.write_text(json.dumps(seeded_precision_payload(args.seed_final, n), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Saved {path} from {args.seed_final}")
        return

    for n in args.n_values:
        result = h.build_estimate(n, formula_orders, args.hncc_orders, args.ncc_orders, q0_by_order, s0_by_order, x_grid)
        payload: dict[str, object] = {
            "metadata": {
                "constraint": "expected_precision",
                "n_values": [n],
                "orders": args.orders,
                "ncc_orders": args.ncc_orders,
                "hncc_orders": args.hncc_orders,
                "q0_by_order": q0_by_order,
                "s0_by_order": s0_by_order,
                "extrap_bases": args.extrap_bases,
                "time": args.time,
                "eps": args.eps,
                "time_values": args.time_values,
                "fixed_eps": args.fixed_eps,
                "x_min": args.x_min,
                "x_max": args.x_max,
                "num_x": args.num_x,
                "scan": args.scan,
            },
            "eps_scan": {},
            "time_scan": {},
        }
        if args.scan in ("eps", "both"):
            payload["eps_scan"] = {
                str(n): precision_rows_for_time(result, args.eps, args.time, args.orders, extrap_formulas)
            }
        if args.scan in ("time", "both"):
            payload["time_scan"] = {
                str(float(time_value)): precision_rows_for_time(
                    result,
                    [args.fixed_eps],
                    time_value,
                    args.orders,
                    extrap_formulas,
                    scan_value=float(time_value),
                )
                for time_value in args.time_values
            }
        path = output_path(args.output_dir, n)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
