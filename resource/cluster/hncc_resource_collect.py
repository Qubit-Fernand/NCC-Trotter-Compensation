#!/usr/bin/env python3
"""Collect HNCC resource-estimate data without drawing figures.

This is the data-only companion to ``hncc_resource_estimate.py``.  It keeps the
same numerical pipeline and command-line defaults, but writes the computed rows
to ``resource/data`` so plotting can be done separately from a notebook.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Mapping, Sequence

import numpy as np

import hncc_resource_estimate as h


def parse_int_map(items: Sequence[str], defaults: Mapping[int, int]) -> Dict[int, int]:
    out = dict(defaults)
    for item in items:
        key, value = item.split(":", 1)
        out[int(key)] = int(value)
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--N", dest="n_values", nargs="+", type=int, default=[6], help="Even system sizes.")
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
    parser.add_argument("--sampling-overhead-cap", type=float, default=math.e, help="Maximum accumulated compensation 1-norm lambda^nu.")
    parser.add_argument("--x-min", type=float, default=1e-3, help="Minimum step size in interpolation grid.")
    parser.add_argument("--x-max", type=float, default=1.0, help="Maximum step size in interpolation grid.")
    parser.add_argument("--num-x", type=int, default=40, help="Number of interpolation-grid points.")
    parser.add_argument("--scan", choices=("eps", "time", "both"), default="both", help="Which scans to collect.")
    parser.add_argument("--data-dir", type=Path, default=Path("resource/data"), help="Output data directory.")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path. Defaults to DATA_DIR/hncc_resource_data_N8_cap_1.75.json.")
    return parser.parse_args()


def json_scalar(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    return value


def rows_to_dicts(rows: Sequence[h.ResourceRow]) -> list[dict[str, object]]:
    return [{key: json_scalar(value) for key, value in asdict(row).items()} for row in rows]


def main() -> None:
    args = parse_args()
    if any(n % 2 for n in args.n_values):
        raise ValueError("Use even N for the periodic even-odd decomposition.")
    if any(order < 2 or order % 2 for order in args.orders + args.ncc_orders + args.hncc_orders + args.extrap_bases):
        raise ValueError("Use even Suzuki orders at least 2.")
    if args.extrap_points != [3]:
        raise ValueError("This simplified comparison keeps only the 3-point extrapolation formula.")
    if args.num_x < 4:
        raise ValueError("--num-x must be at least 4.")

    default_q0 = {order: 2 * order + 2 for order in args.hncc_orders}
    default_s0 = {order: 2 * order + 2 for order in args.hncc_orders}
    q0_by_order = parse_int_map(args.q0, default_q0)
    s0_by_order = parse_int_map(args.s0, default_s0)
    x_grid = np.logspace(math.log10(args.x_min), math.log10(args.x_max), args.num_x)
    extrap_formulas = [h.richardson_formula(base, (1, 2, 6)) for base in args.extrap_bases]
    formula_orders = sorted(set(args.orders) | set(args.extrap_bases) | set(args.ncc_orders) | set(args.hncc_orders))

    results = [
        h.build_estimate(
            n,
            formula_orders,
            args.hncc_orders,
            args.ncc_orders,
            q0_by_order,
            s0_by_order,
            x_grid,
        )
        for n in args.n_values
    ]

    payload: dict[str, object] = {
        "metadata": {
            "n_values": args.n_values,
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
            "sampling_overhead_cap": args.sampling_overhead_cap,
            "x_min": args.x_min,
            "x_max": args.x_max,
            "num_x": args.num_x,
            "scan": args.scan,
        },
        "eps_scan": {},
        "time_scan": {},
    }

    if args.scan in ("eps", "both"):
        eps_scan: dict[str, object] = {}
        for result in results:
            rows = h.resource_rows_for_time(
                result,
                args.eps,
                args.time,
                args.orders,
                extrap_formulas,
                args.sampling_overhead_cap,
            )
            h.print_rows(f"N={result.n}, eps scan at t={args.time:g}", rows)
            eps_scan[str(result.n)] = rows_to_dicts(rows)
        payload["eps_scan"] = eps_scan

    if args.scan in ("time", "both"):
        result = results[0]
        time_scan: dict[str, object] = {}
        for time_value in args.time_values:
            rows = h.resource_rows_for_time(
                result,
                [args.fixed_eps],
                time_value,
                args.orders,
                extrap_formulas,
                args.sampling_overhead_cap,
                scan_value=float(time_value),
            )
            h.print_rows(f"N={result.n}, time scan at eps={args.fixed_eps:.1e}, t={time_value:g}", rows)
            time_scan[str(float(time_value))] = rows_to_dicts(rows)
        payload["time_scan"] = time_scan

    args.data_dir.mkdir(parents=True, exist_ok=True)
    output = args.output if args.output is not None else args.data_dir / "hncc_resource_data_N8_cap_1.75.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
