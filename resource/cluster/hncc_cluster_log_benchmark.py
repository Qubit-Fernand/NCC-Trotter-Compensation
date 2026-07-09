#!/usr/bin/env python3
"""Validate the connected-cluster formal-log backend.

The resource estimator uses the cluster expansion by default.  This small
script keeps the expensive full-system formal log only as a correctness check
on small instances.
"""

from __future__ import annotations

import argparse
import time

import hncc_resource_estimate as h


def print_comparison(full: h.PauliPoly, cluster: h.PauliPoly, qmax: int) -> None:
    print("q   full_norm        cluster_norm     diff_norm        full_terms  cluster_terms")
    print("-------------------------------------------------------------------------------")
    for q in range(1, qmax + 1):
        diff = h.op_add(full.get(q, {}), cluster.get(q, {}), scale_b=-1.0)
        print(
            f"{q:<3d} {h.op_norm1(full.get(q, {})):<16.8g} "
            f"{h.op_norm1(cluster.get(q, {})):<16.8g} "
            f"{h.op_norm1(diff):<16.3e} "
            f"{len(full.get(q, {})):<11d} {len(cluster.get(q, {})):<13d}"
        )


def run_case(n_qubits: int, qmax: int, order: int) -> None:
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

    print_comparison(full, cluster, qmax)
    print(
        f"full formal log time:     {full_time:.4f}s\n"
        f"cluster formal log time:  {cluster_time:.4f}s\n"
        f"representative count:     {representative_count}"
    )
    if full_time > 0:
        print(f"cluster/full ratio:       {cluster_time / full_time:.3g}x")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--N", dest="n_values", nargs="+", type=int, default=[6, 8], help="Even system sizes.")
    parser.add_argument("--qmax", nargs="+", type=int, default=[5, 6], help="BCH truncation orders.")
    parser.add_argument("--order", type=int, default=2, help="Base Suzuki order.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for n_qubits in args.n_values:
        if n_qubits % 2:
            raise ValueError("N must be even for this Heisenberg grouping.")
        for qmax in args.qmax:
            run_case(n_qubits, qmax, args.order)


if __name__ == "__main__":
    main()
