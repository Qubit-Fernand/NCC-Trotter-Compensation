#!/usr/bin/env python3
"""Compute step counts from the accumulated 1-norm constraint only.

For NCC/HNCC rows this solves

    lambda(T / r)^r <= cap.

Plain product formulas and Richardson rows have no r-dependent 1-norm cap in
the current resource model, so they record ``r_1_norm = 1``.
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


class NormOnlyResult:
    def __init__(self, ncc: dict[int, h.NccData], hncc: dict[int, h.HnccData]) -> None:
        self.ncc = ncc
        self.hncc = hncc


def parse_int_map(items: Sequence[str], defaults: Mapping[int, int]) -> dict[int, int]:
    out = dict(defaults)
    for item in items:
        key, value = item.split(":", 1)
        out[int(key)] = int(value)
    return out


def parse_cap(value: str) -> float:
    if value.lower() in {"e", "math.e"}:
        return math.e
    return float(value)


def cap_label(cap: float) -> str:
    if abs(cap - math.e) < 1e-12:
        return "e"
    return f"{cap:g}".replace(".", "p")


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
    parser.add_argument("--time-values", nargs="+", type=float, default=[0.25, 0.35, 0.5, 0.7, 1.0, 1.4, 2.0, 2.8, 4.0], help="Times for time scan.")
    parser.add_argument("--sampling-overhead-cap", nargs="+", type=parse_cap, default=[1.75, math.e], help="1-norm caps, e.g. 1.75 e.")
    parser.add_argument("--x-min", type=float, default=1e-3, help="Minimum step size in interpolation grid.")
    parser.add_argument("--x-max", type=float, default=1.0, help="Maximum step size in interpolation grid.")
    parser.add_argument("--num-x", type=int, default=40, help="Number of interpolation-grid points.")
    parser.add_argument("--output-dir", type=Path, default=Path("resource/data/split_constraints"), help="Output directory.")
    return parser.parse_args()


def min_steps_for_norm(time: float, cap: float, step_lambda_at_x) -> tuple[int, float, float]:
    log_cap = math.log(cap)

    def feasible(steps: int) -> bool:
        x = time / steps
        return steps * math.log(float(step_lambda_at_x(x))) <= log_cap

    hi = 1
    while not feasible(hi):
        hi *= 2
        if hi > 10**8:
            raise RuntimeError(f"could not find 1-norm step count for time={time:g}, cap={cap:g}")
    lo = 0
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if feasible(mid):
            hi = mid
        else:
            lo = mid
    x = time / hi
    norm_overhead = math.exp(hi * math.log(float(step_lambda_at_x(x))))
    return hi, x, norm_overhead


def norm_rows_for_time(
    result: NormOnlyResult,
    time: float,
    formula_orders: Sequence[int],
    extrap_formulas: Sequence[h.ExtrapolationFormula],
    cap: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for order in formula_orders:
        rows.append(row("S" + str(order), time, cap, 1, time, 1.0))
    for formula in extrap_formulas:
        rows.append(row(formula.name, time, cap, 1, time / max(formula.multipliers), formula.norm1))
    for comp in result.ncc.values():
        steps, x, norm_overhead = min_steps_for_norm(
            time,
            cap,
            lambda value, comp=comp: h.ncc_unitary_step_lambda(comp.norms, comp.lead_order, comp.q0, value),
        )
        rows.append(row(comp.method, time, cap, steps, x, norm_overhead))
    for comp in result.hncc.values():
        steps, x, norm_overhead = min_steps_for_norm(
            time,
            cap,
            lambda value, comp=comp: h.eval_poly(comp.norm_poly, value),
        )
        rows.append(row(comp.method, time, cap, steps, x, norm_overhead))
    return rows


def row(method: str, time: float, cap: float, steps: int, x: float, norm_overhead: float) -> dict[str, object]:
    return {
        "method": method,
        "time": float(time),
        "sampling_overhead_cap": float(cap),
        "r_1_norm": int(steps),
        "x_1_norm": float(x),
        "norm_overhead_at_r_1_norm": float(norm_overhead),
    }


def build_norm_only_result(
    n: int,
    hncc_orders: Sequence[int],
    ncc_orders: Sequence[int],
    q0_by_order: Mapping[int, int],
    s0_by_order: Mapping[int, int],
) -> NormOnlyResult:
    a_op, b_op, _ = h.heisenberg_even_odd(n)
    ncc_data: dict[int, h.NccData] = {}
    hncc_data: dict[int, h.HnccData] = {}
    print(f"\n=== N={n} periodic Heisenberg exact-Pauli 1-norm data ===")
    print(f"Pauli terms: A terms={len(a_op)}, B terms={len(b_op)}")
    for order in sorted(set(hncc_orders) | set(ncc_orders)):
        lead = order + 1
        hncc_q0 = q0_by_order[order] if order in hncc_orders else lead
        ncc_q0 = 2 * order + 1 if order in ncc_orders else lead
        qmax = max(hncc_q0, ncc_q0)
        if order in hncc_orders and s0_by_order[order] < lead:
            raise ValueError(f"HNCC{order} requires s0 >= {lead}.")
        if order in hncc_orders and hncc_q0 < lead:
            raise ValueError(f"HNCC{order} requires q0 >= {lead}.")
        print(f"Computing exact Pauli decomposition for order-{order} 1-norm data: qmax={qmax} ...")
        omega_poly = h.remainder_log_pauli_poly(a_op, b_op, order, qmax, n)
        norms_all = {q: h.op_norm1(omega_poly.get(q, {})) for q in range(lead, qmax + 1)}
        print("  exact Pauli 1-norms: " + ", ".join(f"a_{q}={norms_all[q]:.6g}" for q in range(lead, qmax + 1)))
        if order in ncc_orders:
            ncc_q0 = 2 * order + 1
            norms = {q: norms_all[q] for q in range(lead, ncc_q0 + 1)}
            ncc_data[order] = h.NccData(order, lead, ncc_q0, omega_poly, {}, norms, np.array([]))
        if order in hncc_orders:
            q0 = q0_by_order[order]
            s0 = s0_by_order[order]
            norms = {q: norms_all[q] for q in range(lead, q0 + 1)}
            norm_poly = h.hncc_norm_poly(norms, lead, q0, s0)
            print("  HNCC lambda_step(x)-1 terms: " + ", ".join(f"x^{q}:{c:.6g}" for q, c in sorted(norm_poly.items()) if q > 0))
            hncc_data[order] = h.HnccData(order, lead, q0, s0, omega_poly, {}, norms, norm_poly, np.array([]))
    return NormOnlyResult(ncc_data, hncc_data)


def output_path(output_dir: Path, n: int, cap: float) -> Path:
    return output_dir / f"r_for_1_norm_N{n}_cap_{cap_label(cap)}.json"


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
    extrap_formulas = [h.richardson_formula(base, (1, 2, 6)) for base in args.extrap_bases]
    norm_times = sorted({float(args.time), *(float(value) for value in args.time_values)})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for n in args.n_values:
        result = build_norm_only_result(n, args.hncc_orders, args.ncc_orders, q0_by_order, s0_by_order)
        for cap in args.sampling_overhead_cap:
            payload: dict[str, object] = {
                "metadata": {
                    "constraint": "one_norm",
                    "n_values": [n],
                    "orders": args.orders,
                    "ncc_orders": args.ncc_orders,
                    "hncc_orders": args.hncc_orders,
                    "q0_by_order": q0_by_order,
                    "s0_by_order": s0_by_order,
                    "extrap_bases": args.extrap_bases,
                    "time": args.time,
                    "time_values": args.time_values,
                    "sampling_overhead_cap": cap,
                    "x_min": args.x_min,
                    "x_max": args.x_max,
                    "num_x": args.num_x,
                },
                "norm_scan": {
                    str(float(time)): norm_rows_for_time(result, time, args.orders, extrap_formulas, cap)
                    for time in norm_times
                },
            }
            path = output_path(args.output_dir, n, cap)
            path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(f"Saved {path}")


if __name__ == "__main__":
    main()
