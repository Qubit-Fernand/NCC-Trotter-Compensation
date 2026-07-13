#!/usr/bin/env python3
"""Merge split HNCC constraints and compute final resource rows.

The script reads

* ``r_for_expected_precision_N*.json``
* ``r_for_1_norm_N*_cap_*.json``

then uses ``r = max(r_precision, r_1_norm)`` and multiplies by the current
per-step gate-cost model from ``hncc_resource_estimate.py``.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
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


def parse_cap(value: str) -> float:
    if value.lower() in {"e", "math.e"}:
        return math.e
    return float(value)


def cap_label(cap: float) -> str:
    if abs(cap - math.e) < 1e-12:
        return "e"
    return f"{cap:g}".replace(".", "p")


def error_norm_suffix(error_norm: str) -> str:
    return "" if error_norm == "fro" else f"_{error_norm}"


def precision_file_name(n: int, error_norm: str) -> str:
    return f"r_for_expected_precision_N{n}{error_norm_suffix(error_norm)}.json"


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
    parser.add_argument("--sampling-overhead-cap", nargs="+", type=parse_cap, default=[1.75, math.e], help="1-norm caps, e.g. 1.75 e.")
    parser.add_argument("--x-min", type=float, default=1e-3, help="Minimum step size in interpolation grid.")
    parser.add_argument("--x-max", type=float, default=1.0, help="Maximum step size in interpolation grid.")
    parser.add_argument("--num-x", type=int, default=40, help="Number of interpolation-grid points.")
    parser.add_argument("--precision-error-norm", choices=h.ERROR_NORMS, default="trace", help="Error norm used by the expected-precision input file.")
    parser.add_argument("--scan", choices=("eps", "time", "both"), default="both", help="Which scans to merge.")
    parser.add_argument("--input-dir", type=Path, default=Path("resource/data/split_constraints"), help="Directory containing split constraint JSON files.")
    parser.add_argument("--output-dir", type=Path, default=Path("resource/data/split_constraints"), help="Output directory.")
    parser.add_argument(
        "--use-reference-final",
        action="store_true",
        help="Use existing legacy final data for costs/bias and add split provenance fields, avoiding dense step-error recomputation.",
    )
    parser.add_argument(
        "--skip-final-bias-recompute",
        action="store_true",
        help="Use bias_precision from the expected-precision file after merging r=max(r_precision,r_1_norm). Fails if the 1-norm constraint increases r.",
    )
    parser.add_argument("--reference-dir", type=Path, default=Path("resource/data"), help="Directory containing legacy final resource JSON files.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def json_scalar(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    return value


def row_to_dict(row: h.ResourceRow, extras: Mapping[str, object]) -> dict[str, object]:
    out = {key: json_scalar(value) for key, value in asdict(row).items()}
    out.update(extras)
    return out


def norm_index(norm_payload: Mapping[str, object]) -> dict[tuple[str, float], Mapping[str, object]]:
    out: dict[tuple[str, float], Mapping[str, object]] = {}
    for time_key, rows in norm_payload["norm_scan"].items():  # type: ignore[index,union-attr]
        time = float(time_key)
        for row in rows:  # type: ignore[union-attr]
            out[(row["method"], time)] = row
    return out


def active_constraint(r_precision: int, r_norm: int) -> str:
    if r_precision > r_norm:
        return "expected_precision"
    if r_norm > r_precision:
        return "one_norm"
    return "both"


def reference_final_path(reference_dir: Path, n: int, cap: float) -> Path:
    if n == 8 and abs(cap - 1.75) < 1e-12:
        return reference_dir / "hncc_resource_data_N8_cap_1.75.json"
    if n == 8 and abs(cap - math.e) < 1e-12:
        return reference_dir / "hncc_resource_data_N8_cap_e.json"
    if n == 10 and abs(cap - 1.75) < 1e-12:
        return reference_dir / "hncc_resource_data_N10.json"
    if n == 10 and abs(cap - math.e) < 1e-12:
        return reference_dir / "hncc_resource_data_N10_cap_e.json"
    return reference_dir / f"hncc_resource_data_N{n}_cap_{cap_label(cap)}.json"


def add_split_provenance(
    final_payload: Mapping[str, object],
    precision_payload: Mapping[str, object],
    norm_payload: Mapping[str, object],
    n: int,
    cap: float,
) -> dict[str, object]:
    by_precision: dict[tuple[str, str, str, float], Mapping[str, object]] = {}
    for scan_name in ("eps_scan", "time_scan"):
        scan = precision_payload.get(scan_name, {})
        for group, rows in scan.items():  # type: ignore[union-attr]
            for row_data in rows:
                by_precision[(scan_name, str(group), row_data["method"], float(row_data["scan_value"]))] = row_data
    by_norm = norm_index(norm_payload)

    out = json.loads(json.dumps(final_payload))
    metadata = dict(out.get("metadata", {}))
    metadata["source"] = "split_constraints_reference_final"
    metadata["expected_precision_file"] = precision_file_name(n, str(precision_payload.get("metadata", {}).get("error_norm", "fro")))
    metadata["one_norm_file"] = f"r_for_1_norm_N{n}_cap_{cap_label(cap)}.json"
    out["metadata"] = metadata

    for scan_name in ("eps_scan", "time_scan"):
        scan = out.get(scan_name, {})
        for group, rows in scan.items():  # type: ignore[union-attr]
            for row_data in rows:
                key = (scan_name, str(group), row_data["method"], float(row_data["scan_value"]))
                precision_row = by_precision[key]
                norm_row = by_norm[(row_data["method"], float(precision_row["time"]))]
                r_precision = int(precision_row["r_precision"])
                r_norm = int(norm_row["r_1_norm"])
                row_data["r_precision"] = r_precision
                row_data["r_1_norm"] = r_norm
                row_data["active_constraint"] = active_constraint(r_precision, r_norm)
    return out


def comp_norm_overhead(result: h.EstimateResult, method: str, steps: int, x: float) -> float:
    if method.startswith("NCC"):
        order = int(method.removeprefix("NCC"))
        comp = result.ncc[order]
        step_lambda = h.ncc_unitary_step_lambda(comp.norms, comp.lead_order, comp.q0, x)
        return math.exp(steps * math.log(step_lambda))
    if method.startswith("HNCC"):
        order = int(method.removeprefix("HNCC"))
        comp = result.hncc[order]
        return math.exp(steps * math.log(float(h.eval_poly(comp.norm_poly, x))))
    raise ValueError(f"unsupported compensation method {method}")


def comp_bias(result: h.EstimateResult, method: str, steps: int, x: float) -> float:
    if method.startswith("NCC"):
        order = int(method.removeprefix("NCC"))
        return steps * h.log_interpolate_error(result.x_grid, result.ncc[order].step_errors, x)
    if method.startswith("HNCC"):
        order = int(method.removeprefix("HNCC"))
        return steps * h.log_interpolate_error(result.x_grid, result.hncc[order].step_errors, x)
    raise ValueError(f"unsupported compensation method {method}")


def build_resource_skeleton(
    n: int,
    formula_orders: Sequence[int],
    hncc_orders: Sequence[int],
    ncc_orders: Sequence[int],
    q0_by_order: Mapping[int, int],
    s0_by_order: Mapping[int, int],
    x_grid: np.ndarray,
    error_norm: str,
) -> h.EstimateResult:
    a_op, b_op, _ = h.heisenberg_even_odd(n)
    formula_orders = tuple(sorted(set(formula_orders) | set(hncc_orders) | set(ncc_orders)))
    formula_layers = {order: h.suzuki_layers(order) for order in formula_orders}
    ncc_data: dict[int, h.NccData] = {}
    hncc_data: dict[int, h.HnccData] = {}
    for order in sorted(set(hncc_orders) | set(ncc_orders)):
        lead = order + 1
        hncc_q0 = q0_by_order[order] if order in hncc_orders else lead
        ncc_q0 = 2 * order + 1 if order in ncc_orders else lead
        qmax = max(hncc_q0, ncc_q0)
        omega_poly = h.remainder_log_pauli_poly(a_op, b_op, order, qmax, n)
        norms_all = {q: h.op_norm1(omega_poly.get(q, {})) for q in range(lead, qmax + 1)}
        if order in ncc_orders:
            ncc_norms = {q: norms_all[q] for q in range(lead, ncc_q0 + 1)}
            ncc_data[order] = h.NccData(order, lead, ncc_q0, omega_poly, {}, ncc_norms, np.array([]))
        if order in hncc_orders:
            q0 = q0_by_order[order]
            s0 = s0_by_order[order]
            hncc_norms = {q: norms_all[q] for q in range(lead, q0 + 1)}
            norm_poly = h.hncc_norm_poly(hncc_norms, lead, q0, s0)
            hncc_data[order] = h.HnccData(order, lead, q0, s0, omega_poly, {}, hncc_norms, norm_poly, np.array([]))
    return h.EstimateResult(
        n=n,
        a_op=a_op,
        b_op=b_op,
        a_eigh=(np.array([]), np.array([[]])),
        b_eigh=(np.array([]), np.array([[]])),
        h_eigh=(np.array([]), np.array([[]])),
        formula_layers=formula_layers,
        formula_errors={},
        hncc=hncc_data,
        ncc=ncc_data,
        x_grid=x_grid,
        error_norm=error_norm,
    )


def final_row(
    result: h.EstimateResult,
    precision_row: Mapping[str, object],
    norm_row: Mapping[str, object],
    formula_by_name: Mapping[str, h.ExtrapolationFormula],
    a_cost: h.GateCost,
    b_cost: h.GateCost,
) -> dict[str, object]:
    method = str(precision_row["method"])
    eps = float(precision_row["eps"])
    time = float(precision_row["time"])
    r_precision = int(precision_row["r_precision"])
    r_norm = int(norm_row["r_1_norm"])
    steps = max(r_precision, r_norm)
    constraint = active_constraint(r_precision, r_norm)

    if method.startswith("S"):
        order = int(method.removeprefix("S"))
        x = time / steps
        if steps == r_precision:
            bias = float(precision_row["bias_precision"])
        else:
            if order not in result.formula_errors:
                raise RuntimeError(f"{method} needs final-bias recomputation at r={steps}, but no error curve is available.")
            bias = steps * h.log_interpolate_error(result.x_grid, result.formula_errors[order], x)
        norm_overhead = 1.0
        cost = h.layer_sequence_gate_cost(result.formula_layers[order], steps, a_cost, b_cost)
        row = h.ResourceRow(
            float(precision_row["scan_value"]),
            method,
            steps,
            x,
            bias,
            norm_overhead,
            cost.total,
            cost.total,
            cost.cnot,
            cost.cnot,
            cost.rz,
            cost.rz,
            eps**-2,
        )
    elif method.startswith("RE"):
        formula = formula_by_name[method]
        x = time / (steps * max(formula.multipliers))
        # Richardson rows have no r-dependent 1-norm cap in the current model,
        # so r_final should equal r_precision.  Reuse the cached precision bias.
        bias = float(precision_row["bias_precision"])
        max_steps = steps * max(formula.multipliers)
        cost = h.layer_sequence_gate_cost(result.formula_layers[formula.base_order], max_steps, a_cost, b_cost)
        row = h.ResourceRow(
            float(precision_row["scan_value"]),
            method,
            steps,
            x,
            bias,
            formula.norm1,
            cost.total,
            cost.total,
            cost.cnot,
            cost.cnot,
            cost.rz,
            cost.rz,
            (eps**-2) * formula.points * (formula.norm1**2),
        )
    elif method.startswith("NCC"):
        order = int(method.removeprefix("NCC"))
        comp = result.ncc[order]
        x = time / steps
        if steps == r_precision:
            bias = float(precision_row["bias_precision"])
        else:
            if len(comp.step_errors) == 0:
                raise RuntimeError(f"{method} needs final-bias recomputation at r={steps}, but no error curve is available.")
            bias = comp_bias(result, method, steps, x)
        norm_overhead = comp_norm_overhead(result, method, steps, x)
        formula_step = h.one_step_formula_gate_cost(result.formula_layers[comp.base_order], a_cost, b_cost)
        correction = h.ncc_correction_gate_stats(comp, x)
        row = h.ResourceRow(
            float(precision_row["scan_value"]),
            method,
            steps,
            x,
            bias,
            norm_overhead,
            steps * (formula_step.total + 2.0 * (correction.expected_cnot + correction.expected_rz)),
            steps * (formula_step.total + 2.0 * (correction.max_cnot + correction.max_rz)),
            steps * (formula_step.cnot + 2.0 * correction.expected_cnot),
            steps * (formula_step.cnot + 2.0 * correction.max_cnot),
            steps * (formula_step.rz + 2.0 * correction.expected_rz),
            steps * (formula_step.rz + 2.0 * correction.max_rz),
            (eps**-2) * (norm_overhead**4),
            correction.expected_pauli_weight,
            correction.max_pauli_weight,
        )
    elif method.startswith("HNCC"):
        order = int(method.removeprefix("HNCC"))
        comp = result.hncc[order]
        x = time / steps
        if steps == r_precision:
            bias = float(precision_row["bias_precision"])
        else:
            if len(comp.step_errors) == 0:
                raise RuntimeError(f"{method} needs final-bias recomputation at r={steps}, but no error curve is available.")
            bias = comp_bias(result, method, steps, x)
        norm_overhead = comp_norm_overhead(result, method, steps, x)
        formula_step = h.one_step_formula_gate_cost(result.formula_layers[comp.base_order], a_cost, b_cost)
        correction = h.correction_gate_stats(comp, x)
        row = h.ResourceRow(
            float(precision_row["scan_value"]),
            method,
            steps,
            x,
            bias,
            norm_overhead,
            steps * (formula_step.total + correction.expected_cnot + correction.expected_rz),
            steps * (formula_step.total + correction.max_cnot + correction.max_rz),
            steps * (formula_step.cnot + correction.expected_cnot),
            steps * (formula_step.cnot + correction.max_cnot),
            steps * (formula_step.rz + correction.expected_rz),
            steps * (formula_step.rz + correction.max_rz),
            (eps**-2) * (norm_overhead**2),
            correction.expected_pauli_weight,
            correction.max_pauli_weight,
        )
    else:
        raise ValueError(f"unsupported method {method}")

    return row_to_dict(
        row,
        {
            "r_precision": r_precision,
            "r_1_norm": r_norm,
            "active_constraint": constraint,
        },
    )


def output_path(output_dir: Path, n: int, cap: float, error_norm: str) -> Path:
    return output_dir / f"hncc_resource_split_N{n}_cap_{cap_label(cap)}{error_norm_suffix(error_norm)}.json"


def main() -> None:
    args = parse_args()
    if any(n % 2 for n in args.n_values):
        raise ValueError("Use even N for the periodic even-odd decomposition.")
    if args.extrap_points != [3]:
        raise ValueError("This simplified comparison keeps only the 3-point extrapolation formula.")
    if args.use_reference_final and args.precision_error_norm != "fro":
        raise ValueError("--use-reference-final is only valid for the legacy Frobenius precision data.")

    default_q0 = {order: 2 * order + 2 for order in args.hncc_orders}
    default_s0 = {order: 2 * order + 2 for order in args.hncc_orders}
    q0_by_order = parse_int_map(args.q0, default_q0)
    s0_by_order = parse_int_map(args.s0, default_s0)
    x_grid = np.logspace(math.log10(args.x_min), math.log10(args.x_max), args.num_x)
    extrap_formulas = [h.richardson_formula(base, (1, 2, 6)) for base in args.extrap_bases]
    formula_by_name = {formula.name: formula for formula in extrap_formulas}
    formula_orders = sorted(set(args.orders) | set(args.extrap_bases) | set(args.ncc_orders) | set(args.hncc_orders))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for n in args.n_values:
        precision_payload = load_json(args.input_dir / precision_file_name(n, args.precision_error_norm))
        precision_metadata = precision_payload.get("metadata", {})
        if precision_metadata.get("error_norm", "fro") != args.precision_error_norm:
            raise ValueError(
                f"precision file for N={n} has error_norm={precision_metadata.get('error_norm', 'fro')!r}, "
                f"but requested {args.precision_error_norm!r}."
            )
        result: h.EstimateResult | None = None
        a_cost: h.GateCost | None = None
        b_cost: h.GateCost | None = None

        for cap in args.sampling_overhead_cap:
            norm_payload = load_json(args.input_dir / f"r_for_1_norm_N{n}_cap_{cap_label(cap)}.json")
            if args.use_reference_final:
                final_payload = load_json(reference_final_path(args.reference_dir, n, cap))
                payload = add_split_provenance(final_payload, precision_payload, norm_payload, n, cap)
                path = output_path(args.output_dir, n, cap, args.precision_error_norm)
                path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
                print(f"Saved {path} from {reference_final_path(args.reference_dir, n, cap)}")
                continue

            if result is None:
                if args.skip_final_bias_recompute:
                    result = build_resource_skeleton(n, formula_orders, args.hncc_orders, args.ncc_orders, q0_by_order, s0_by_order, x_grid, args.precision_error_norm)
                else:
                    result = h.build_estimate(n, formula_orders, args.hncc_orders, args.ncc_orders, q0_by_order, s0_by_order, x_grid, args.precision_error_norm)
                a_cost = h.op_rotation_gate_cost(result.a_op)
                b_cost = h.op_rotation_gate_cost(result.b_op)
            assert a_cost is not None and b_cost is not None
            by_norm = norm_index(norm_payload)
            payload: dict[str, object] = {
                "metadata": {
                    "source": "split_constraints",
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
                    "error_norm": args.precision_error_norm,
                    "sampling_overhead_cap": cap,
                    "x_min": args.x_min,
                    "x_max": args.x_max,
                    "num_x": args.num_x,
                    "scan": args.scan,
                    "expected_precision_file": precision_file_name(n, args.precision_error_norm),
                    "one_norm_file": f"r_for_1_norm_N{n}_cap_{cap_label(cap)}.json",
                },
                "eps_scan": {},
                "time_scan": {},
            }
            if args.scan in ("eps", "both"):
                eps_rows = precision_payload["eps_scan"][str(n)]  # type: ignore[index]
                payload["eps_scan"] = {
                    str(n): [
                        final_row(result, row, by_norm[(row["method"], float(row["time"]))], formula_by_name, a_cost, b_cost)
                        for row in eps_rows
                    ]
                }
            if args.scan in ("time", "both"):
                time_scan: dict[str, object] = {}
                for time_key, rows in precision_payload["time_scan"].items():  # type: ignore[index,union-attr]
                    time_scan[time_key] = [
                        final_row(result, row, by_norm[(row["method"], float(row["time"]))], formula_by_name, a_cost, b_cost)
                        for row in rows
                    ]
                payload["time_scan"] = time_scan
            path = output_path(args.output_dir, n, cap, args.precision_error_norm)
            path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            print(f"Saved {path}")


if __name__ == "__main__":
    main()
