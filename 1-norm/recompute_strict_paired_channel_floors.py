#!/usr/bin/env python3
"""Recompute strict-paired channel 1-norm floors for the K=1 data tables.

The calculation uses the exact sparse Pauli decomposition of Phi_2, Phi_3,
and Phi_4 for the periodic Heisenberg chain.  For x=T/r it imposes

  original:  [1 + 2(a2 x^2 + a3 x^3)^2]^r <= cap,
  HNCC:      [1 + 2(a2 x^2 + a3 x^3 + a4 x^4)^2
                + 2 a2^2 x^4]^r <= cap.

The HNCC expression pairs every single Phi_q (q=2,3,4); the only Taylor-tail
term through s0=4 is (ad_{Phi_2})^2 / 2!.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections.abc import MutableMapping
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESOURCE_CLUSTER = ROOT / "resource" / "cluster"
sys.path.insert(0, str(RESOURCE_CLUSTER))

from hncc_resource_estimate import (  # noqa: E402
    PauliOp,
    add_term,
    clean_op,
    exp_poly_from_order1,
    log_poly_near_identity,
    op_add,
    op_identity,
    op_norm1,
    op_scale,
    pauli_term,
    poly_mul,
)


def channel_heisenberg_ab(n: int) -> tuple[PauliOp, PauliOp, PauliOp]:
    """Sparse equivalent of scaling.Pauli_Hamiltonian_BCH.build_periodic_ab.

    Unlike the resource-estimation helper, the channel data includes odd N in
    the sweep, so this intentionally follows the channel layer assignment
    literally rather than requiring an even bipartition.
    """
    a_terms: MutableMapping[tuple[int, ...], complex] = {}
    b_terms: MutableMapping[tuple[int, ...], complex] = {}
    for site in range(n):
        target = a_terms if site % 2 == 0 else b_terms
        nxt = (site + 1) % n
        for label in (1, 2, 3):
            add_term(target, pauli_term(n, [site, nxt], [label, label]))
        add_term(target, pauli_term(n, [site], [3]))
    a_op, b_op = clean_op(a_terms), clean_op(b_terms)
    return a_op, b_op, op_add(a_op, b_op)


def phi_norms(n: int) -> tuple[float, float, float]:
    """Return exact Pauli 1-norms of Phi_2, Phi_3, Phi_4.

    This computes the formal logarithm of
    exp(i B x) exp(i A x) exp(-i(A+B)x) through x^4 in sparse Pauli algebra.
    It is equivalent to ``scaling/Pauli_Hamiltonian_BCH.phi_term`` but avoids
    its dense 4^N Pauli-basis scan.
    """
    a_op, b_op, h_op = channel_heisenberg_ab(n)
    qmax = 4
    product = {0: op_identity(n)}
    for generator in (op_scale(b_op, 1j), op_scale(a_op, 1j), op_scale(h_op, -1j)):
        product = poly_mul(product, exp_poly_from_order1(generator, qmax, n), qmax)
    phi = log_poly_near_identity(product, qmax, n)
    return tuple(op_norm1(phi[q]) for q in (2, 3, 4))  # type: ignore[return-value]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cap", type=float, default=1.75)
    parser.add_argument("--n-values", type=int, nargs="+", default=[4, 5, 6, 7, 8])
    parser.add_argument("--times", type=float, nargs="+", default=[0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
    parser.add_argument("--time-for-n-sweep", type=float, default=2.0)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    return parser.parse_args()


def minimum_steps(time: float, cap: float, one_step_norm) -> int:
    if cap <= 1:
        raise ValueError("cap must exceed one")
    r = 1
    log_cap = math.log(cap)
    while r * math.log(one_step_norm(time / r)) > log_cap:
        r += 1
    return r


def one_step_norms(a2: float, a3: float, a4: float):
    def original(x: float) -> float:
        return 1.0 + 2.0 * (a2 * x**2 + a3 * x**3) ** 2

    def hncc(x: float) -> float:
        pair_eta = a2 * x**2 + a3 * x**3 + a4 * x**4
        return 1.0 + 2.0 * pair_eta**2 + 2.0 * a2**2 * x**4

    return original, hncc


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    coefficient_n_values = sorted(set(args.n_values) | {8})
    coefficients = {n: phi_norms(n) for n in coefficient_n_values}
    for n, (a2, a3, a4) in coefficients.items():
        print(f"N={n}: a2={a2:.12g}, a3={a3:.12g}, a4={a4:.12g}")

    n8 = coefficients[8]
    original8, hncc8 = one_step_norms(*n8)
    time_rows = [
        {
            "T": f"{time:g}",
            "original_operator": "",
            "original_channel": minimum_steps(time, args.cap, original8),
            "log_operator_s6": "",
            "log_channel_s4": minimum_steps(time, args.cap, hncc8),
        }
        for time in args.times
    ]
    n_rows = []
    for n in args.n_values:
        original, hncc = one_step_norms(*coefficients[n])
        n_rows.append(
            {
                "N": n,
                "original_operator": "",
                "log_operator_s6": "",
                "original_channel": minimum_steps(args.time_for_n_sweep, args.cap, original),
                "log_channel_s4": minimum_steps(args.time_for_n_sweep, args.cap, hncc),
            }
        )

    label = f"{args.cap:g}".replace(".", "p")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(
        args.output_dir / f"pauli_sampling_overhead_rmin_N8_cap_{label}.csv",
        ["T", "original_operator", "original_channel", "log_operator_s6", "log_channel_s4"],
        time_rows,
    )
    write_csv(
        args.output_dir / f"pauli_sampling_overhead_rmin_N_sweep_T2_cap_{label}.csv",
        ["N", "original_operator", "log_operator_s6", "original_channel", "log_channel_s4"],
        n_rows,
    )


if __name__ == "__main__":
    main()
