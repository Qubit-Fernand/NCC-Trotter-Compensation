#!/usr/bin/env python3
"""Collect connected-cluster formal-log benchmark data.

This is the data-only companion to ``hncc_cluster_log_benchmark.py``.  It keeps
the full-system formal log only as a small-instance correctness check and writes
the comparison data to ``scripts/data`` for notebook-based plotting.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import hncc_resource_estimate as h


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--N", dest="n_values", nargs="+", type=int, default=[6, 8], help="Even system sizes.")
    parser.add_argument("--qmax", nargs="+", type=int, default=[5, 6], help="BCH truncation orders.")
    parser.add_argument("--order", type=int, default=2, help="Base Suzuki order.")
    parser.add_argument("--data-dir", type=Path, default=Path("scripts/data"), help="Output data directory.")
    parser.add_argument("--output", type=Path, default=None, help="Output JSON path. Defaults to DATA_DIR/hncc_cluster_log_data.json.")
    return parser.parse_args()


def collect_case(n_qubits: int, qmax: int, order: int) -> dict[str, Any]:
    a_op, b_op, _ = h.heisenberg_even_odd(n_qubits)
    print(f"\n=== N={n_qubits}, S{order}, qmax={qmax} ===")
    print(f"A terms={len(a_op)}, B terms={len(b_op)}")

    start = time.perf_counter()
    full = h.full_system_remainder_log_pauli_poly(a_op, b_op, order, qmax, n_qubits)
    full_time = time.perf_counter() - start

    start = time.perf_counter()
    cluster, representative_count = h.connected_cluster_remainder_log_pauli_poly(
        a_op,
        b_op,
        order,
        qmax,
        n_qubits,
    )
    cluster_time = time.perf_counter() - start

    orders = []
    print("q   full_norm        cluster_norm     diff_norm        full_terms  cluster_terms")
    print("-------------------------------------------------------------------------------")
    for q in range(1, qmax + 1):
        diff = h.op_add(full.get(q, {}), cluster.get(q, {}), scale_b=-1.0)
        row = {
            "q": q,
            "full_norm": h.op_norm1(full.get(q, {})),
            "cluster_norm": h.op_norm1(cluster.get(q, {})),
            "diff_norm": h.op_norm1(diff),
            "full_terms": len(full.get(q, {})),
            "cluster_terms": len(cluster.get(q, {})),
        }
        orders.append(row)
        print(
            f"{q:<3d} {row['full_norm']:<16.8g} "
            f"{row['cluster_norm']:<16.8g} "
            f"{row['diff_norm']:<16.3e} "
            f"{row['full_terms']:<11d} {row['cluster_terms']:<13d}"
        )

    print(
        f"full formal log time:     {full_time:.4f}s\n"
        f"cluster formal log time:  {cluster_time:.4f}s\n"
        f"representative count:     {representative_count}"
    )
    if full_time > 0:
        print(f"cluster/full ratio:       {cluster_time / full_time:.3g}x")

    return {
        "N": n_qubits,
        "order": order,
        "qmax": qmax,
        "a_terms": len(a_op),
        "b_terms": len(b_op),
        "full_time": full_time,
        "cluster_time": cluster_time,
        "cluster_full_ratio": cluster_time / full_time if full_time > 0 else None,
        "representative_count": representative_count,
        "orders": orders,
    }


def main() -> None:
    args = parse_args()
    if any(n % 2 for n in args.n_values):
        raise ValueError("N must be even for this Heisenberg grouping.")

    cases = []
    for n_qubits in args.n_values:
        for qmax in args.qmax:
            cases.append(collect_case(n_qubits, qmax, args.order))

    payload = {
        "metadata": {
            "n_values": args.n_values,
            "qmax": args.qmax,
            "order": args.order,
        },
        "cases": cases,
    }

    args.data_dir.mkdir(parents=True, exist_ok=True)
    output = args.output if args.output is not None else args.data_dir / "hncc_cluster_log_data.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Saved {output}")


if __name__ == "__main__":
    main()
