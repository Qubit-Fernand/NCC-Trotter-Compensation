#!/usr/bin/env python3
"""Finite-size HNCC resource estimates from exact Pauli decompositions.

The script uses sparse Pauli-string algebra and a translation-cached connected
cluster formal log to compute the truncated Pauli decomposition of the Trotter
remainder for periodic 1D Heisenberg chains.  It then converts that explicit
Pauli representation into the LCQC normalization used by the HNCC compensation
step and compares resource estimates for

  * plain Suzuki-Trotter formulas S2, S4, S6,
  * unitary-level NCC compensation of S2,
  * HNCC compensation of S2,
  * Richardson / multiproduct extrapolation based on S2 and optionally S4.

No Monte Carlo sampling or shot noise is simulated.  Gate counts are estimated
with the standard cost model for a weight-w Pauli rotation:
2(w-1) CNOT gates and one Rz gate.  NCC compensation uses controlled Pauli
rotations, which are decomposed into 2w CNOT gates and two ordinary Rz gates.
Classical repetitions are reported as epsilon^{-2} for Trotter formulas,
lambda^{2 nu} epsilon^{-2} for HNCC, lambda^{4 nu} epsilon^{-2} for
unitary-level NCC, and m ||a||_1^2 epsilon^{-2} for an m-point extrapolation
formula.
"""

from __future__ import annotations

import argparse
import itertools
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


PauliString = Tuple[int, ...]
PauliOp = Dict[PauliString, complex]
PauliPoly = Dict[int, PauliOp]
Layer = Tuple[str, float]
EighData = Tuple[np.ndarray, np.ndarray]
Cluster = FrozenSet[int]
PauliAction = Tuple[np.ndarray, np.ndarray]

PAULI_MATS = (
    np.array([[1, 0], [0, 1]], dtype=complex),
    np.array([[0, 1], [1, 0]], dtype=complex),
    np.array([[0, -1j], [1j, 0]], dtype=complex),
    np.array([[1, 0], [0, -1]], dtype=complex),
)

SINGLE_MUL = {
    (0, 0): (1, 0),
    (0, 1): (1, 1),
    (0, 2): (1, 2),
    (0, 3): (1, 3),
    (1, 0): (1, 1),
    (2, 0): (1, 2),
    (3, 0): (1, 3),
    (1, 1): (1, 0),
    (2, 2): (1, 0),
    (3, 3): (1, 0),
    (1, 2): (1j, 3),
    (2, 3): (1j, 1),
    (3, 1): (1j, 2),
    (2, 1): (-1j, 3),
    (3, 2): (-1j, 1),
    (1, 3): (-1j, 2),
}


def identity_string(n: int) -> PauliString:
    return (0,) * n


def clean_op(op: Mapping[PauliString, complex], tol: float = 1e-13) -> PauliOp:
    return {p: c for p, c in op.items() if abs(c) > tol}


def op_add(
    a: Mapping[PauliString, complex],
    b: Mapping[PauliString, complex],
    scale_b: complex = 1.0,
) -> PauliOp:
    out: MutableMapping[PauliString, complex] = dict(a)
    for p, c in b.items():
        out[p] = out.get(p, 0.0) + scale_b * c
    return clean_op(out)


def op_scale(a: Mapping[PauliString, complex], scale: complex) -> PauliOp:
    if scale == 0:
        return {}
    return clean_op({p: scale * c for p, c in a.items()})


def multiply_strings(a: PauliString, b: PauliString) -> Tuple[complex, PauliString]:
    phase = 1.0 + 0.0j
    out: List[int] = []
    for pa, pb in zip(a, b):
        local_phase, local_label = SINGLE_MUL[(pa, pb)]
        phase *= local_phase
        out.append(local_label)
    return phase, tuple(out)


def op_mul(a: Mapping[PauliString, complex], b: Mapping[PauliString, complex]) -> PauliOp:
    if not a or not b:
        return {}
    out: MutableMapping[PauliString, complex] = {}
    for pa, ca in a.items():
        for pb, cb in b.items():
            phase, pc = multiply_strings(pa, pb)
            out[pc] = out.get(pc, 0.0) + ca * cb * phase
    return clean_op(out)


def op_norm1(a: Mapping[PauliString, complex]) -> float:
    return float(sum(abs(c) for c in a.values()))


def pauli_weight(p: PauliString) -> int:
    return sum(label != 0 for label in p)


def op_identity(n: int) -> PauliOp:
    return {identity_string(n): 1.0 + 0.0j}


def poly_add(a: PauliPoly, b: PauliPoly, scale_b: complex = 1.0, qmax: int | None = None) -> PauliPoly:
    out: PauliPoly = {q: dict(op) for q, op in a.items()}
    for q, op in b.items():
        if qmax is not None and q > qmax:
            continue
        out[q] = op_add(out.get(q, {}), op, scale_b=scale_b)
        if not out[q]:
            del out[q]
    return out


def poly_scale(a: PauliPoly, scale: complex) -> PauliPoly:
    out: PauliPoly = {}
    for q, op in a.items():
        scaled = op_scale(op, scale)
        if scaled:
            out[q] = scaled
    return out


def poly_mul(a: PauliPoly, b: PauliPoly, qmax: int) -> PauliPoly:
    out: PauliPoly = {}
    for qa, opa in a.items():
        for qb, opb in b.items():
            q = qa + qb
            if q > qmax:
                continue
            prod = op_mul(opa, opb)
            if prod:
                out[q] = op_add(out.get(q, {}), prod)
    return out


def exp_poly_from_order1(base: PauliOp, qmax: int, n_qubits: int) -> PauliPoly:
    out: PauliPoly = {0: op_identity(n_qubits)}
    power = op_identity(n_qubits)
    factorial = 1
    for q in range(1, qmax + 1):
        power = op_mul(power, base)
        factorial *= q
        out[q] = op_scale(power, 1.0 / factorial)
    return out


def log_poly_near_identity(prod: PauliPoly, qmax: int, n_qubits: int) -> PauliPoly:
    y = poly_add(prod, {0: op_identity(n_qubits)}, scale_b=-1, qmax=qmax)
    out: PauliPoly = {}
    y_power = y
    for m in range(1, qmax + 1):
        coeff = ((-1) ** (m + 1)) / m
        out = poly_add(out, poly_scale(y_power, coeff), qmax=qmax)
        if m < qmax:
            y_power = poly_mul(y_power, y, qmax)
            if not y_power:
                break
    return out


def pauli_term(n: int, positions: Sequence[int], labels: Sequence[int], coeff: complex = 1.0) -> PauliOp:
    p = [0] * n
    for pos, label in zip(positions, labels):
        p[pos] = label
    return {tuple(p): coeff}


def add_term(target: MutableMapping[PauliString, complex], term: Mapping[PauliString, complex]) -> None:
    for p, c in term.items():
        target[p] = target.get(p, 0.0) + c


def heisenberg_even_odd(n: int, j_coupling: float = 1.0, h_field: float = 1.0) -> Tuple[PauliOp, PauliOp, PauliOp]:
    """Periodic Heisenberg H=A+B with odd/even bond layers in 1-based indexing."""
    if n % 2 != 0:
        raise ValueError("The even-odd periodic decomposition assumes even N.")
    a_terms: MutableMapping[PauliString, complex] = {}
    b_terms: MutableMapping[PauliString, complex] = {}
    for site in range(n):
        nxt = (site + 1) % n
        layer = a_terms if site % 2 == 0 else b_terms
        add_term(layer, pauli_term(n, [site, nxt], [1, 1], j_coupling))
        add_term(layer, pauli_term(n, [site, nxt], [2, 2], j_coupling))
        add_term(layer, pauli_term(n, [site, nxt], [3, 3], j_coupling))
        add_term(layer, pauli_term(n, [site], [3], h_field))
    a_op = clean_op(a_terms)
    b_op = clean_op(b_terms)
    return a_op, b_op, op_add(a_op, b_op)


def compress_layers(layers: Sequence[Layer], tol: float = 1e-14) -> List[Layer]:
    compressed: List[Layer] = []
    for label, coeff in layers:
        if abs(coeff) <= tol:
            continue
        if compressed and compressed[-1][0] == label:
            new_coeff = compressed[-1][1] + coeff
            if abs(new_coeff) <= tol:
                compressed.pop()
            else:
                compressed[-1] = (label, new_coeff)
        else:
            compressed.append((label, coeff))
    return compressed


def scale_layers(layers: Sequence[Layer], scale: float) -> List[Layer]:
    return [(label, scale * coeff) for label, coeff in layers]


def suzuki_layers(order: int) -> List[Layer]:
    """Grouped A/B layer sequence for the recursive Suzuki formula S_order(x)."""
    if order == 2:
        return [("A", 0.5), ("B", 1.0), ("A", 0.5)]
    if order < 2 or order % 2:
        raise ValueError("Suzuki order must be an even integer at least 2.")
    prev = suzuki_layers(order - 2)
    p_order = 1.0 / (4.0 - 4.0 ** (1.0 / (order - 1)))
    return compress_layers(
        scale_layers(prev, p_order)
        + scale_layers(prev, p_order)
        + scale_layers(prev, 1.0 - 4.0 * p_order)
        + scale_layers(prev, p_order)
        + scale_layers(prev, p_order)
    )


def remainder_bases(a_op: PauliOp, b_op: PauliOp, formula_order: int) -> List[PauliOp]:
    """Return B_j for V(x)=exp(B_0 x) exp(B_1 x) ... = U(x) S(x)^dagger."""
    h_op = op_add(a_op, b_op)
    bases = [op_scale(h_op, -1j)]
    for label, coeff in reversed(suzuki_layers(formula_order)):
        op = a_op if label == "A" else b_op
        bases.append(op_scale(op, 1j * coeff))
    return bases


def full_system_remainder_log_pauli_poly(
    a_op: PauliOp,
    b_op: PauliOp,
    formula_order: int,
    qmax: int,
    n_qubits: int,
) -> PauliPoly:
    """Full-system formal Pauli-algebra log through order qmax."""
    prod: PauliPoly = {0: op_identity(n_qubits)}
    for base in remainder_bases(a_op, b_op, formula_order):
        prod = poly_mul(prod, exp_poly_from_order1(base, qmax, n_qubits), qmax)
    return log_poly_near_identity(prod, qmax, n_qubits)


@dataclass
class ClusterResult:
    connected: PauliPoly


def pauli_support(pauli: PauliString) -> Cluster:
    return frozenset(idx for idx, label in enumerate(pauli) if label != 0)


def restrict_op_to_cluster(op: Mapping[PauliString, complex], cluster: Cluster) -> PauliOp:
    return {pauli: coeff for pauli, coeff in op.items() if pauli_support(pauli).issubset(cluster)}


def cyclic_interval(start: int, length: int, n_qubits: int) -> Cluster:
    if length == n_qubits:
        return frozenset(range(n_qubits))
    return frozenset((start + offset) % n_qubits for offset in range(length))


def poly_subtract(target: PauliPoly, source: PauliPoly, qmax: int) -> PauliPoly:
    return poly_add(target, source, scale_b=-1.0, qmax=qmax)


def poly_accumulate(target: PauliPoly, source: PauliPoly, qmax: int) -> PauliPoly:
    return poly_add(target, source, qmax=qmax)


def translate_pauli(pauli: PauliString, shift: int) -> PauliString:
    n_qubits = len(pauli)
    out = [0] * n_qubits
    for site, label in enumerate(pauli):
        out[(site + shift) % n_qubits] = label
    return tuple(out)


def translate_op(op: Mapping[PauliString, complex], shift: int) -> PauliOp:
    if not op:
        return {}
    n_qubits = len(next(iter(op)))
    shift %= n_qubits
    if shift == 0:
        return dict(op)
    out: PauliOp = {}
    for pauli, coeff in op.items():
        translated = translate_pauli(pauli, shift)
        out[translated] = out.get(translated, 0.0) + coeff
    return clean_op(out)


def translate_poly(poly: PauliPoly, shift: int) -> PauliPoly:
    return {order: translate_op(op, shift) for order, op in poly.items()}


def connected_cluster_remainder_log_pauli_poly(
    a_op: PauliOp,
    b_op: PauliOp,
    formula_order: int,
    qmax: int,
    n_qubits: int,
) -> Tuple[PauliPoly, int]:
    """Connected-interval cluster expansion for the BCH/log remainder.

    The implementation is specialized to the periodic 1D nearest-neighbor
    Heisenberg benchmark.  For a q-fold connected term, the support fits in a
    cyclic interval of length at most q+1.  Even-odd product-formula groupings
    are invariant under translations by two sites, so each interval length has
    at most two representatives.
    """
    max_cluster_length = min(n_qubits, qmax + 1)
    representatives: Dict[Tuple[int, int | None], ClusterResult] = {}
    total_connected: PauliPoly = {}

    for length in range(1, max_cluster_length + 1):
        parity_options: Sequence[int | None] = (None,) if length == n_qubits else (0, 1)
        for parity in parity_options:
            start = 0 if parity is None else parity
            cluster = cyclic_interval(start, length, n_qubits)
            local_a = restrict_op_to_cluster(a_op, cluster)
            local_b = restrict_op_to_cluster(b_op, cluster)
            omega = full_system_remainder_log_pauli_poly(local_a, local_b, formula_order, qmax, n_qubits)
            connected = {order: dict(op) for order, op in omega.items()}

            if length == n_qubits:
                for sub_length in range(1, n_qubits):
                    for sub_start in range(n_qubits):
                        sub_parity = sub_start % 2
                        sub_connected = representatives[(sub_length, sub_parity)].connected
                        connected = poly_subtract(
                            connected,
                            translate_poly(sub_connected, sub_start - sub_parity),
                            qmax,
                        )
            else:
                for sub_length in range(1, length):
                    for offset in range(length - sub_length + 1):
                        sub_start = (start + offset) % n_qubits
                        sub_parity = sub_start % 2
                        sub_connected = representatives[(sub_length, sub_parity)].connected
                        connected = poly_subtract(
                            connected,
                            translate_poly(sub_connected, sub_start - sub_parity),
                            qmax,
                        )

            representatives[(length, parity)] = ClusterResult(connected=connected)

            if length == n_qubits:
                total_connected = poly_accumulate(total_connected, connected, qmax)
            else:
                assert parity is not None
                for target_start in range(parity, n_qubits, 2):
                    total_connected = poly_accumulate(
                        total_connected,
                        translate_poly(connected, target_start - parity),
                        qmax,
                    )

    return total_connected, len(representatives)


def remainder_log_pauli_poly(
    a_op: PauliOp,
    b_op: PauliOp,
    formula_order: int,
    qmax: int,
    n_qubits: int,
) -> PauliPoly:
    """Default BCH/log backend: translation-cached connected clusters."""
    start = time.perf_counter()
    omega_poly, representative_count = connected_cluster_remainder_log_pauli_poly(
        a_op,
        b_op,
        formula_order,
        qmax,
        n_qubits,
    )
    elapsed = time.perf_counter() - start
    print(f"  cluster formal log: representatives={representative_count}, elapsed={elapsed:.3f}s")
    return omega_poly


def pauli_matrix(p: PauliString) -> np.ndarray:
    mat = np.array([[1]], dtype=complex)
    for label in p:
        mat = np.kron(mat, PAULI_MATS[label])
    return mat


def pauli_action(p: PauliString) -> PauliAction:
    """Return row indices and phases for the sparse action of a Pauli string."""
    n = len(p)
    dim = 2**n
    cols = np.arange(dim, dtype=np.int64)
    rows = cols.copy()
    phases = np.ones(dim, dtype=complex)
    for site, label in enumerate(p):
        if label == 0:
            continue
        bit_mask = 1 << (n - 1 - site)
        bit_is_one = (cols & bit_mask) != 0
        if label == 1:
            rows ^= bit_mask
        elif label == 2:
            phases *= np.where(bit_is_one, -1j, 1j)
            rows ^= bit_mask
        elif label == 3:
            phases *= np.where(bit_is_one, -1.0, 1.0)
        else:
            raise ValueError(f"invalid Pauli label {label}")
    return rows, phases


def op_to_matrix(op: Mapping[PauliString, complex], n_qubits: int, cache: MutableMapping[PauliString, PauliAction] | None = None) -> np.ndarray:
    dim = 2**n_qubits
    mat = np.zeros((dim, dim), dtype=complex)
    cols = np.arange(dim, dtype=np.int64)
    for p, c in op.items():
        if cache is None:
            rows, phases = pauli_action(p)
        else:
            if p not in cache:
                cache[p] = pauli_action(p)
            rows, phases = cache[p]
        mat[rows, cols] += c * phases
    return mat


def unitary_from_eigh(eigh_data: EighData, x: float, scale: complex) -> np.ndarray:
    vals, vecs = eigh_data
    return (vecs * np.exp(scale * x * vals)) @ vecs.conj().T


def product_formula_matrix(layers: Sequence[Layer], x: float, a_eigh: EighData, b_eigh: EighData) -> np.ndarray:
    dim = len(a_eigh[0])
    out = np.eye(dim, dtype=complex)
    for label, coeff in layers:
        eigh_data = a_eigh if label == "A" else b_eigh
        out = out @ unitary_from_eigh(eigh_data, x, -1j * coeff)
    return out


def max_basis_fro_error(u_mat: np.ndarray, approx_mat: np.ndarray) -> float:
    overlaps = np.sum(approx_mat.conj() * u_mat, axis=0)
    return float(np.sqrt(np.maximum(2.0 - 2.0 * np.abs(overlaps) ** 2, 0.0)).max())


def commutator(a: np.ndarray, rho: np.ndarray) -> np.ndarray:
    return a @ rho - rho @ a


def ordered_tuples_with_sum(values: Sequence[int], max_sum: int) -> Iterable[Tuple[int, ...]]:
    max_len = max_sum // min(values)
    for length in range(1, max_len + 1):
        for tup in itertools.product(values, repeat=length):
            if sum(tup) <= max_sum:
                yield tup


def apply_hncc_map(
    rho: np.ndarray,
    omega_mats: Mapping[int, np.ndarray],
    x: float,
    lead_order: int,
    q0: int,
    s0: int,
) -> np.ndarray:
    """Apply I + truncated Taylor series of ad_{Omega<=q0} through total order s0."""
    orders = tuple(range(lead_order, q0 + 1))
    out = rho.copy()
    for tup in ordered_tuples_with_sum(orders, s0):
        term = rho
        for order in reversed(tup):
            term = commutator((x**order) * omega_mats[order], term)
        out = out + term / math.factorial(len(tup))
    return out


def hncc_step_error(
    x: float,
    u_mat: np.ndarray,
    s_mat: np.ndarray,
    omega_mats: Mapping[int, np.ndarray],
    lead_order: int,
    q0: int,
    s0: int,
) -> float:
    max_err = 0.0
    dim = u_mat.shape[0]
    for z in range(dim):
        psi_s = s_mat[:, z]
        psi_u = u_mat[:, z]
        rho_s = np.outer(psi_s, psi_s.conj())
        rho_u = np.outer(psi_u, psi_u.conj())
        rho_comp = apply_hncc_map(rho_s, omega_mats, x, lead_order, q0, s0)
        max_err = max(max_err, float(np.linalg.norm(rho_comp - rho_u, ord="fro")))
    return max_err


def ncc_compensation_operator(
    omega_mats: Mapping[int, np.ndarray],
    x: float,
    lead_order: int,
    q0: int,
) -> np.ndarray:
    dim = next(iter(omega_mats.values())).shape[0]
    out = np.eye(dim, dtype=complex)
    for order in range(lead_order, q0 + 1):
        out = out + (x**order) * omega_mats[order]
    return out


def ncc_step_error(
    x: float,
    u_mat: np.ndarray,
    s_mat: np.ndarray,
    omega_mats: Mapping[int, np.ndarray],
    lead_order: int,
    q0: int,
) -> float:
    """Channel error induced by unitary-level NCC cross-term estimation."""
    comp = ncc_compensation_operator(omega_mats, x, lead_order, q0)
    approx_op = comp @ s_mat
    max_err = 0.0
    dim = u_mat.shape[0]
    for z in range(dim):
        psi_u = u_mat[:, z]
        psi_comp = approx_op[:, z]
        rho_u = np.outer(psi_u, psi_u.conj())
        rho_comp = np.outer(psi_comp, psi_comp.conj())
        max_err = max(max_err, float(np.linalg.norm(rho_comp - rho_u, ord="fro")))
    return max_err


def eval_poly(coeffs: Mapping[int, float], x: np.ndarray | float) -> np.ndarray | float:
    return sum(c * (x**q) for q, c in coeffs.items())


def hncc_norm_poly(norms: Mapping[int, float], lead_order: int, q0: int, s0: int) -> Dict[int, float]:
    """One-step LCQC normalization polynomial from exact Pauli 1-norms.

    The single-commutator identity-pairing contribution is modeled as
    1 + 2(lambda_single)^2.  The remaining Taylor terms use the product of the
    exact Pauli 1-norms and the Taylor prefactor.
    """
    coeffs: Dict[int, float] = {0: 1.0}
    for q1 in range(lead_order, q0 + 1):
        for q2 in range(lead_order, q0 + 1):
            coeffs[q1 + q2] = coeffs.get(q1 + q2, 0.0) + 2.0 * norms.get(q1, 0.0) * norms.get(q2, 0.0)

    for tup in ordered_tuples_with_sum(tuple(range(lead_order, q0 + 1)), s0):
        if len(tup) <= 1:
            continue
        order = sum(tup)
        weight = 1.0 / math.factorial(len(tup))
        for q in tup:
            weight *= 2.0 * norms.get(q, 0.0)
        coeffs[order] = coeffs.get(order, 0.0) + weight
    return {q: c for q, c in coeffs.items() if abs(c) > 0}


def log_interpolate_error(xs: np.ndarray, ys: np.ndarray, x: float, floor: float = 1e-14) -> float:
    valid = np.flatnonzero(np.isfinite(ys) & (ys > floor))
    if len(valid) < 3:
        valid = np.flatnonzero(np.isfinite(ys) & (ys > 0))
    if len(valid) < 3:
        raise RuntimeError("not enough positive error data for interpolation")

    xv = xs[valid]
    yv = ys[valid]
    logx = np.log(xv)
    logy = np.log(yv)
    lx = math.log(x)
    if logx[0] <= lx <= logx[-1]:
        return float(math.exp(np.interp(lx, logx, logy)))

    if lx < logx[0]:
        chosen = slice(0, min(8, len(logx)))
    else:
        chosen = slice(max(0, len(logx) - 8), len(logx))
    slope, intercept = np.polyfit(logx[chosen], logy[chosen], deg=1)
    return float(math.exp(slope * lx + intercept))


def rotation_gate_cost(weight: int) -> Tuple[int, int]:
    if weight <= 0:
        return 0, 0
    return 2 * (weight - 1), 1


def controlled_rotation_gate_cost(weight: int) -> Tuple[int, int]:
    if weight <= 0:
        return 0, 0
    # Compute and uncompute the Pauli parity. The controlled-Rz is decomposed
    # into two CNOT gates and two ordinary Rz rotations.
    return 2 * weight, 2


@dataclass
class GateCost:
    cnot: float
    rz: float

    @property
    def total(self) -> float:
        return self.cnot + self.rz


def op_rotation_gate_cost(op: Mapping[PauliString, complex]) -> GateCost:
    cnot = 0
    rz = 0
    for p in op:
        local_cnot, local_rz = rotation_gate_cost(pauli_weight(p))
        cnot += local_cnot
        rz += local_rz
    return GateCost(float(cnot), float(rz))


def layer_sequence_gate_cost(layers: Sequence[Layer], steps: int, a_cost: GateCost, b_cost: GateCost) -> GateCost:
    repeated_layers = compress_layers(list(layers) * steps)
    cnot = 0.0
    rz = 0.0
    for label, _ in repeated_layers:
        cost = a_cost if label == "A" else b_cost
        cnot += cost.cnot
        rz += cost.rz
    return GateCost(cnot, rz)


def one_step_formula_gate_cost(layers: Sequence[Layer], a_cost: GateCost, b_cost: GateCost) -> GateCost:
    cnot = 0.0
    rz = 0.0
    for label, _ in layers:
        cost = a_cost if label == "A" else b_cost
        cnot += cost.cnot
        rz += cost.rz
    return GateCost(cnot, rz)


@dataclass
class PauliWeightStats:
    expected_pauli_weight: float
    max_pauli_weight: int
    expected_cnot: float
    max_cnot: int
    expected_rz: float
    max_rz: int


def weighted_pauli_stats(
    omega_poly: PauliPoly,
    orders: Iterable[int],
    x: float,
    controlled: bool = False,
) -> PauliWeightStats:
    total = 0.0
    weight_sum = 0.0
    cnot_sum = 0.0
    rz_sum = 0.0
    max_weight = 0
    max_cnot = 0
    max_rz = 0
    for q in orders:
        for p, coeff in omega_poly.get(q, {}).items():
            sample_weight = abs(coeff) * (x**q)
            if sample_weight == 0:
                continue
            weight = pauli_weight(p)
            cost_fn = controlled_rotation_gate_cost if controlled else rotation_gate_cost
            cnot, rz = cost_fn(weight)
            total += sample_weight
            weight_sum += sample_weight * weight
            cnot_sum += sample_weight * cnot
            rz_sum += sample_weight * rz
            max_weight = max(max_weight, weight)
            max_cnot = max(max_cnot, cnot)
            max_rz = max(max_rz, rz)
    if total == 0:
        return PauliWeightStats(0.0, 0, 0.0, 0, 0.0, 0)
    return PauliWeightStats(
        expected_pauli_weight=weight_sum / total,
        max_pauli_weight=max_weight,
        expected_cnot=cnot_sum / total,
        max_cnot=max_cnot,
        expected_rz=rz_sum / total,
        max_rz=max_rz,
    )


@dataclass
class CorrectionGateStats:
    expected_pauli_weight: float
    max_pauli_weight: int
    expected_cnot: float
    max_cnot: int
    expected_rz: float
    max_rz: int


def ncc_unitary_step_lambda(norms: Mapping[int, float], lead_order: int, q0: int, x: float) -> float:
    """Unitary-level identity-paired LCU norm for I + sum_q x^q Omega_q.

    If Lambda=sum |c_l| for the Pauli terms in the compensation operator, then
    I + sum_l c_l iP_l = sqrt(1+Lambda^2) E_l[exp(theta_l iP_l)] with
    tan(theta_l)=sign(c_l) Lambda.  This is the unitary-level analogue of
    identity pairing and has no factor of 2 from the adjoint action.
    """
    lambda_single = sum(norms.get(q, 0.0) * x**q for q in range(lead_order, q0 + 1))
    return math.sqrt(1.0 + lambda_single**2)


def ncc_correction_gate_stats(comp: "NccData", x: float) -> CorrectionGateStats:
    orders = range(comp.lead_order, comp.q0 + 1)
    stats = weighted_pauli_stats(comp.omega_poly, orders, x, controlled=True)
    return CorrectionGateStats(
        expected_pauli_weight=stats.expected_pauli_weight,
        max_pauli_weight=stats.max_pauli_weight,
        expected_cnot=stats.expected_cnot,
        max_cnot=stats.max_cnot,
        expected_rz=stats.expected_rz,
        max_rz=stats.max_rz,
    )


def correction_gate_stats(comp: "HnccData", x: float) -> CorrectionGateStats:
    single_orders = range(comp.lead_order, comp.q0 + 1)
    single = weighted_pauli_stats(comp.omega_poly, single_orders, x)
    lambda_single = sum(comp.norms.get(q, 0.0) * x**q for q in single_orders)
    lambda_single_pair = 1.0 + 2.0 * lambda_single**2

    branch_weights = [lambda_single_pair]
    # The identity-paired single branch has total norm 1+2 lambda_single^2,
    # but only the non-identity weight 2 lambda_single^2 carries a Pauli
    # rotation gate.
    branch_expected_cnot = [(2.0 * lambda_single**2 / lambda_single_pair) * single.expected_cnot]
    branch_expected_rz = [(2.0 * lambda_single**2 / lambda_single_pair) * single.expected_rz]
    branch_expected_weight = [(2.0 * lambda_single**2 / lambda_single_pair) * single.expected_pauli_weight]
    max_cnot = single.max_cnot
    max_rz = single.max_rz
    max_weight = single.max_pauli_weight

    orders = tuple(range(comp.lead_order, comp.q0 + 1))
    for tup in ordered_tuples_with_sum(orders, comp.s0):
        if len(tup) <= 1:
            continue
        branch_lambda = 1.0 / math.factorial(len(tup))
        expected_cnot = 0.0
        expected_rz = 0.0
        expected_weight = 0.0
        tuple_max_cnot = 0
        tuple_max_rz = 0
        tuple_max_weight = 0
        for q in tup:
            branch_lambda *= 2.0 * comp.norms.get(q, 0.0) * x**q
            stats = weighted_pauli_stats(comp.omega_poly, [q], x)
            expected_cnot += stats.expected_cnot
            expected_rz += stats.expected_rz
            expected_weight += stats.expected_pauli_weight
            tuple_max_cnot += stats.max_cnot
            tuple_max_rz += stats.max_rz
            tuple_max_weight += stats.max_pauli_weight
        branch_weights.append(branch_lambda)
        branch_expected_cnot.append(expected_cnot)
        branch_expected_rz.append(expected_rz)
        branch_expected_weight.append(expected_weight)
        max_cnot = max(max_cnot, tuple_max_cnot)
        max_rz = max(max_rz, tuple_max_rz)
        max_weight = max(max_weight, tuple_max_weight)

    lambda_step = sum(branch_weights)
    expected_cnot = sum(w * c for w, c in zip(branch_weights, branch_expected_cnot)) / lambda_step
    expected_rz = sum(w * r for w, r in zip(branch_weights, branch_expected_rz)) / lambda_step
    expected_weight = sum(w * wt for w, wt in zip(branch_weights, branch_expected_weight)) / lambda_step
    return CorrectionGateStats(expected_weight, max_weight, expected_cnot, max_cnot, expected_rz, max_rz)


@dataclass(frozen=True)
class ExtrapolationFormula:
    name: str
    base_order: int
    multipliers: Tuple[int, ...]
    coeffs: Tuple[float, ...]

    @property
    def norm1(self) -> float:
        return float(sum(abs(c) for c in self.coeffs))

    @property
    def points(self) -> int:
        return len(self.coeffs)


def richardson_formula(base_order: int, multipliers: Sequence[int]) -> ExtrapolationFormula:
    powers = [0] + [base_order * j for j in range(1, len(multipliers))]
    mat = np.array([[m ** (-p) for m in multipliers] for p in powers], dtype=float)
    rhs = np.zeros(len(multipliers), dtype=float)
    rhs[0] = 1.0
    coeffs = tuple(float(c) for c in np.linalg.solve(mat, rhs))
    name = f"RE{base_order}-{len(multipliers)}"
    return ExtrapolationFormula(name, base_order, tuple(multipliers), coeffs)


def extrapolation_channel_error(
    formula: ExtrapolationFormula,
    macro_steps: int,
    time: float,
    h_eigh: EighData,
    a_eigh: EighData,
    b_eigh: EighData,
    layers_by_order: Mapping[int, List[Layer]],
) -> float:
    exact = unitary_from_eigh(h_eigh, time, -1j)
    base_layers = layers_by_order[formula.base_order]
    approx_unitaries = []
    for multiplier in formula.multipliers:
        total_steps = macro_steps * multiplier
        step = product_formula_matrix(base_layers, time / total_steps, a_eigh, b_eigh)
        approx_unitaries.append(np.linalg.matrix_power(step, total_steps))

    max_error = 0.0
    dim = exact.shape[0]
    for z in range(dim):
        psi_exact = exact[:, z]
        rho_exact = np.outer(psi_exact, psi_exact.conj())
        rho_extrap = np.zeros_like(rho_exact)
        for coeff, approx in zip(formula.coeffs, approx_unitaries):
            psi = approx[:, z]
            rho_extrap += coeff * np.outer(psi, psi.conj())
        max_error = max(max_error, float(np.linalg.norm(rho_extrap - rho_exact, ord="fro")))
    return max_error


@dataclass
class HnccData:
    base_order: int
    lead_order: int
    q0: int
    s0: int
    omega_poly: PauliPoly
    omega_mats: Dict[int, np.ndarray]
    norms: Dict[int, float]
    norm_poly: Dict[int, float]
    step_errors: np.ndarray

    @property
    def method(self) -> str:
        return f"HNCC{self.base_order}"


@dataclass
class NccData:
    base_order: int
    lead_order: int
    q0: int
    omega_poly: PauliPoly
    omega_mats: Dict[int, np.ndarray]
    norms: Dict[int, float]
    step_errors: np.ndarray

    @property
    def method(self) -> str:
        return f"NCC{self.base_order}"


@dataclass
class EstimateResult:
    n: int
    a_op: PauliOp
    b_op: PauliOp
    a_eigh: EighData
    b_eigh: EighData
    h_eigh: EighData
    formula_layers: Dict[int, List[Layer]]
    formula_errors: Dict[int, np.ndarray]
    hncc: Dict[int, HnccData]
    ncc: Dict[int, NccData]
    x_grid: np.ndarray


@dataclass
class ResourceRow:
    scan_value: float
    method: str
    steps: int
    x: float
    bias: float
    norm_overhead: float
    expected_gates: float
    max_gates: float
    expected_cnot: float
    max_cnot: float
    expected_rz: float
    max_rz: float
    repetitions: float
    expected_comp_weight: float = 0.0
    max_comp_weight: int = 0


def find_min_formula_steps(result: EstimateResult, formula_order: int, eps: float, time: float) -> Tuple[int, float, float]:
    errors = result.formula_errors[formula_order]

    def feasible(steps: int) -> bool:
        x = time / steps
        return steps * log_interpolate_error(result.x_grid, errors, x) <= eps

    hi = 1
    while not feasible(hi):
        hi *= 2
        if hi > 10**8:
            raise RuntimeError(f"could not find feasible step count for S{formula_order}, eps={eps:g}")
    lo = 0
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if feasible(mid):
            hi = mid
        else:
            lo = mid
    x = time / hi
    return hi, x, hi * log_interpolate_error(result.x_grid, errors, x)


def find_min_hncc_steps(comp: HnccData, x_grid: np.ndarray, eps: float, time: float, overhead_cap: float) -> Tuple[int, float, float, float]:
    def feasible(steps: int) -> bool:
        x = time / steps
        step_lambda = float(eval_poly(comp.norm_poly, x))
        log_overhead = steps * math.log(step_lambda)
        if log_overhead > math.log(overhead_cap):
            return False
        return steps * log_interpolate_error(x_grid, comp.step_errors, x) <= eps

    hi = 1
    while not feasible(hi):
        hi *= 2
        if hi > 10**8:
            raise RuntimeError(f"could not find feasible step count for {comp.method}, eps={eps:g}")
    lo = 0
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if feasible(mid):
            hi = mid
        else:
            lo = mid
    x = time / hi
    bias = hi * log_interpolate_error(x_grid, comp.step_errors, x)
    norm_overhead = math.exp(hi * math.log(float(eval_poly(comp.norm_poly, x))))
    return hi, x, bias, norm_overhead


def find_min_ncc_steps(comp: NccData, x_grid: np.ndarray, eps: float, time: float, overhead_cap: float) -> Tuple[int, float, float, float]:
    def feasible(steps: int) -> bool:
        x = time / steps
        step_lambda = ncc_unitary_step_lambda(comp.norms, comp.lead_order, comp.q0, x)
        log_overhead = steps * math.log(step_lambda)
        if log_overhead > math.log(overhead_cap):
            return False
        return steps * log_interpolate_error(x_grid, comp.step_errors, x) <= eps

    hi = 1
    while not feasible(hi):
        hi *= 2
        if hi > 10**8:
            raise RuntimeError(f"could not find feasible step count for {comp.method}, eps={eps:g}")
    lo = 0
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if feasible(mid):
            hi = mid
        else:
            lo = mid
    x = time / hi
    bias = hi * log_interpolate_error(x_grid, comp.step_errors, x)
    step_lambda = ncc_unitary_step_lambda(comp.norms, comp.lead_order, comp.q0, x)
    norm_overhead = math.exp(hi * math.log(step_lambda))
    return hi, x, bias, norm_overhead


def find_min_extrapolation_steps(
    result: EstimateResult,
    formula: ExtrapolationFormula,
    eps: float,
    time: float,
) -> Tuple[int, float, float]:
    cache: Dict[int, float] = {}

    def error(macro_steps: int) -> float:
        if macro_steps not in cache:
            cache[macro_steps] = extrapolation_channel_error(
                formula,
                macro_steps,
                time,
                result.h_eigh,
                result.a_eigh,
                result.b_eigh,
                result.formula_layers,
            )
        return cache[macro_steps]

    hi = 1
    while error(hi) > eps:
        hi *= 2
        if hi > 10**7:
            raise RuntimeError(f"could not find feasible step count for {formula.name}, eps={eps:g}")
    lo = 0
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if error(mid) <= eps:
            hi = mid
        else:
            lo = mid
    return hi, time / (hi * max(formula.multipliers)), error(hi)


def build_estimate(
    n: int,
    formula_orders: Sequence[int],
    hncc_orders: Sequence[int],
    ncc_orders: Sequence[int],
    q0_by_order: Mapping[int, int],
    s0_by_order: Mapping[int, int],
    x_grid: np.ndarray,
) -> EstimateResult:
    a_op, b_op, h_op = heisenberg_even_odd(n)
    gamma = len(h_op)
    formula_orders = tuple(sorted(set(formula_orders) | set(hncc_orders) | set(ncc_orders)))
    formula_layers = {order: suzuki_layers(order) for order in formula_orders}

    cache: Dict[PauliString, np.ndarray] = {}
    a_mat = op_to_matrix(a_op, n, cache)
    b_mat = op_to_matrix(b_op, n, cache)
    h_mat = a_mat + b_mat
    a_eigh = np.linalg.eigh(a_mat)
    b_eigh = np.linalg.eigh(b_mat)
    h_eigh = np.linalg.eigh(h_mat)

    formula_errors: Dict[int, List[float]] = {order: [] for order in formula_orders}
    for x in x_grid:
        exact = unitary_from_eigh(h_eigh, x, -1j)
        for order in formula_orders:
            approx = product_formula_matrix(formula_layers[order], x, a_eigh, b_eigh)
            formula_errors[order].append(max_basis_fro_error(exact, approx))
    formula_error_arrays = {order: np.array(vals) for order, vals in formula_errors.items()}

    hncc_data: Dict[int, HnccData] = {}
    ncc_data: Dict[int, NccData] = {}
    print(f"\n=== N={n} periodic Heisenberg exact-Pauli resource data ===")
    print(f"Pauli terms: Gamma={gamma}, A terms={len(a_op)}, B terms={len(b_op)}")
    for order in sorted(set(hncc_orders) | set(ncc_orders)):
        lead = order + 1
        hncc_q0 = q0_by_order[order] if order in hncc_orders else lead
        ncc_q0 = 2 * order + 1 if order in ncc_orders else lead
        qmax = max(hncc_q0, ncc_q0)
        if order in hncc_orders and s0_by_order[order] < lead:
            raise ValueError(f"HNCC{order} requires s0 >= {lead}.")
        if order in hncc_orders and hncc_q0 < lead:
            raise ValueError(f"HNCC{order} requires q0 >= {lead}.")
        print(f"Computing exact Pauli decomposition for order-{order} compensation: qmax={qmax} ...")
        omega_poly = remainder_log_pauli_poly(a_op, b_op, order, qmax, n)
        norms_all = {q: op_norm1(omega_poly.get(q, {})) for q in range(lead, qmax + 1)}
        below = {q: op_norm1(omega_poly.get(q, {})) for q in range(1, lead)}
        print("  lower-order checks: " + ", ".join(f"Omega_{q}={v:.3e}" for q, v in below.items()))
        print("  exact Pauli 1-norms: " + ", ".join(f"a_{q}={norms_all[q]:.6g}" for q in range(lead, qmax + 1)))
        omega_mats_by_q: Dict[int, np.ndarray] = {}

        def omega_mat(q: int) -> np.ndarray:
            if q not in omega_mats_by_q:
                omega_mats_by_q[q] = op_to_matrix(omega_poly.get(q, {}), n)
            return omega_mats_by_q[q]

        if order in ncc_orders:
            norms = {q: norms_all[q] for q in range(lead, ncc_q0 + 1)}
            omega_mats = {q: omega_mat(q) for q in range(lead, ncc_q0 + 1)}
            step_errors = []
            for x in x_grid:
                exact = unitary_from_eigh(h_eigh, x, -1j)
                s_mat = product_formula_matrix(formula_layers[order], x, a_eigh, b_eigh)
                step_errors.append(ncc_step_error(x, exact, s_mat, omega_mats, lead, ncc_q0))
            print(f"  NCC{order}: compensating orders {lead} through {ncc_q0}")
            ncc_data[order] = NccData(order, lead, ncc_q0, omega_poly, omega_mats, norms, np.array(step_errors))

        if order in hncc_orders:
            q0 = q0_by_order[order]
            s0 = s0_by_order[order]
            norms = {q: norms_all[q] for q in range(lead, q0 + 1)}
            omega_mats = {q: omega_mat(q) for q in range(lead, q0 + 1)}
            step_errors = []
            for x in x_grid:
                exact = unitary_from_eigh(h_eigh, x, -1j)
                s_mat = product_formula_matrix(formula_layers[order], x, a_eigh, b_eigh)
                step_errors.append(hncc_step_error(x, exact, s_mat, omega_mats, lead, q0, s0))
            norm_poly = hncc_norm_poly(norms, lead, q0, s0)
            print(f"  HNCC{order}: q0={q0}, s0={s0}")
            print("  HNCC lambda_step(x)-1 terms: " + ", ".join(f"x^{q}:{c:.6g}" for q, c in sorted(norm_poly.items()) if q > 0))
            hncc_data[order] = HnccData(order, lead, q0, s0, omega_poly, omega_mats, norms, norm_poly, np.array(step_errors))

    return EstimateResult(
        n=n,
        a_op=a_op,
        b_op=b_op,
        a_eigh=a_eigh,
        b_eigh=b_eigh,
        h_eigh=h_eigh,
        formula_layers=formula_layers,
        formula_errors=formula_error_arrays,
        hncc=hncc_data,
        ncc=ncc_data,
        x_grid=x_grid,
    )


def resource_rows_for_time(
    result: EstimateResult,
    eps_values: Sequence[float],
    time: float,
    formula_orders: Sequence[int],
    extrap_formulas: Sequence[ExtrapolationFormula],
    overhead_cap: float,
    scan_value: float | None = None,
) -> List[ResourceRow]:
    a_cost = op_rotation_gate_cost(result.a_op)
    b_cost = op_rotation_gate_cost(result.b_op)
    rows: List[ResourceRow] = []

    for eps in eps_values:
        row_scan_value = eps if scan_value is None else scan_value
        for order in formula_orders:
            steps, x, bias = find_min_formula_steps(result, order, eps, time)
            cost = layer_sequence_gate_cost(result.formula_layers[order], steps, a_cost, b_cost)
            rows.append(
                ResourceRow(
                    row_scan_value,
                    f"S{order}",
                    steps,
                    x,
                    bias,
                    1.0,
                    cost.total,
                    cost.total,
                    cost.cnot,
                    cost.cnot,
                    cost.rz,
                    cost.rz,
                    eps ** -2,
                )
            )

        for formula in extrap_formulas:
            macro_steps, x, bias = find_min_extrapolation_steps(result, formula, eps, time)
            max_steps = macro_steps * max(formula.multipliers)
            cost = layer_sequence_gate_cost(result.formula_layers[formula.base_order], max_steps, a_cost, b_cost)
            repetitions = (eps ** -2) * formula.points * (formula.norm1**2)
            rows.append(
                ResourceRow(
                    row_scan_value,
                    formula.name,
                    macro_steps,
                    x,
                    bias,
                    formula.norm1,
                    cost.total,
                    cost.total,
                    cost.cnot,
                    cost.cnot,
                    cost.rz,
                    cost.rz,
                    repetitions,
                )
            )

        for comp in result.ncc.values():
            steps, x, bias, norm_overhead = find_min_ncc_steps(comp, result.x_grid, eps, time, overhead_cap)
            formula_step = one_step_formula_gate_cost(result.formula_layers[comp.base_order], a_cost, b_cost)
            correction = ncc_correction_gate_stats(comp, x)
            # Unit-level NCC estimates L rho L^\dagger, so the cross-term
            # estimator samples one compensation unitary on each side.
            expected = steps * (formula_step.total + 2.0 * (correction.expected_cnot + correction.expected_rz))
            worst = steps * (formula_step.total + 2.0 * (correction.max_cnot + correction.max_rz))
            expected_cnot = steps * (formula_step.cnot + 2.0 * correction.expected_cnot)
            worst_cnot = steps * (formula_step.cnot + 2.0 * correction.max_cnot)
            expected_rz = steps * (formula_step.rz + 2.0 * correction.expected_rz)
            worst_rz = steps * (formula_step.rz + 2.0 * correction.max_rz)
            repetitions = (eps ** -2) * (norm_overhead**4)
            rows.append(
                ResourceRow(
                    row_scan_value,
                    comp.method,
                    steps,
                    x,
                    bias,
                    norm_overhead,
                    expected,
                    worst,
                    expected_cnot,
                    worst_cnot,
                    expected_rz,
                    worst_rz,
                    repetitions,
                    correction.expected_pauli_weight,
                    correction.max_pauli_weight,
                )
            )

        for comp in result.hncc.values():
            steps, x, bias, norm_overhead = find_min_hncc_steps(comp, result.x_grid, eps, time, overhead_cap)
            formula_step = one_step_formula_gate_cost(result.formula_layers[comp.base_order], a_cost, b_cost)
            correction = correction_gate_stats(comp, x)
            expected = steps * (formula_step.total + correction.expected_cnot + correction.expected_rz)
            worst = steps * (formula_step.total + correction.max_cnot + correction.max_rz)
            expected_cnot = steps * (formula_step.cnot + correction.expected_cnot)
            worst_cnot = steps * (formula_step.cnot + correction.max_cnot)
            expected_rz = steps * (formula_step.rz + correction.expected_rz)
            worst_rz = steps * (formula_step.rz + correction.max_rz)
            repetitions = (eps ** -2) * (norm_overhead**2)
            rows.append(
                ResourceRow(
                    row_scan_value,
                    comp.method,
                    steps,
                    x,
                    bias,
                    norm_overhead,
                    expected,
                    worst,
                    expected_cnot,
                    worst_cnot,
                    expected_rz,
                    worst_rz,
                    repetitions,
                    correction.expected_pauli_weight,
                    correction.max_pauli_weight,
                )
            )
    return rows


def print_rows(title: str, rows: Sequence[ResourceRow]) -> None:
    print(f"\n=== {title} ===")
    header = (
        "scan       method    steps       x          bias       norm/||a|| "
        "max CNOT   max Rz     repetitions  E[Wc] maxW"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row.scan_value:<10.3g} {row.method:<8} {row.steps:<10d} {row.x:<10.4g} "
            f"{row.bias:<10.3e} {row.norm_overhead:<10.3g} "
            f"{row.max_cnot:<10.3g} {row.max_rz:<10.3g} {row.repetitions:<11.3g} "
            f"{row.expected_comp_weight:<5.2f} {row.max_comp_weight:<4d}"
        )


def unique_legend(axs: Sequence[plt.Axes]) -> Tuple[List[object], List[str]]:
    by_label: Dict[str, object] = {}
    for ax in axs:
        handles, labels = ax.get_legend_handles_labels()
        for handle, label in zip(handles, labels):
            by_label.setdefault(label, handle)
    return list(by_label.values()), list(by_label.keys())


T_SYNTHESIS_EPS = 1e-15
T_SYNTHESIS_COST = 1.149 * math.log2(1.0 / T_SYNTHESIS_EPS) + 9.2


def max_t_gates(row: ResourceRow) -> float:
    return row.max_rz * T_SYNTHESIS_COST


def make_eps_tradeoff_plot(
    all_rows: Mapping[int, Sequence[ResourceRow]],
    eps_values: Sequence[float],
    out_dir: Path,
) -> None:
    fig, (ax_cnot, ax_t, ax_rep) = plt.subplots(1, 3, figsize=(10.8, 3.2))
    eps_sorted = np.array(sorted(eps_values), dtype=float)
    n_count = len(all_rows)
    for n, rows in all_rows.items():
        gate_methods = sorted({row.method for row in rows}, key=method_sort_key)
        gate_by_method = {method: {row.scan_value: row for row in rows if row.method == method} for method in gate_methods}
        for method in gate_methods:
            cnot_values = np.array([gate_by_method[method][eps].max_cnot for eps in eps_sorted])
            t_values = np.array([max_t_gates(gate_by_method[method][eps]) for eps in eps_sorted])
            ax_cnot.loglog(
                eps_sorted,
                cnot_values,
                marker_for_method(method),
                ms=3.4,
                label=legend_label(n, method, n_count),
                **line_style_for_method(method),
            )
            ax_t.loglog(
                eps_sorted,
                t_values,
                marker_for_method(method),
                ms=3.4,
                label=legend_label(n, method, n_count),
                **line_style_for_method(method),
            )

        rep_methods = sorted({repetition_display_method(row.method) for row in rows}, key=method_sort_key)
        rep_by_method: Dict[str, Dict[float, ResourceRow]] = {method: {} for method in rep_methods}
        for row in rows:
            method = repetition_display_method(row.method)
            rep_by_method[method].setdefault(row.scan_value, row)
        for method in rep_methods:
            values = np.array([rep_by_method[method][eps].repetitions * eps**2 for eps in eps_sorted])
            ax_rep.semilogx(
                eps_sorted,
                values,
                marker_for_method(method),
                ms=3.4,
                label=legend_label(n, method, n_count),
                **line_style_for_method(method),
            )

    ax_cnot.set_ylabel(r"Max CNOT gates per run")
    ax_t.set_ylabel(r"Max $T$ gates per run")
    ax_rep.set_ylabel(r"Repetition overhead $N_{\rm rep}\epsilon^2$")
    for ax in (ax_cnot, ax_t, ax_rep):
        ax.set_xlabel(r"Target precision $\epsilon$")
        ax.invert_xaxis()
        ax.grid(True, which="both", ls=":", lw=0.55)
        ax.tick_params(axis="both", labelsize=8.5)
        ax.xaxis.label.set_size(10)
        ax.yaxis.label.set_size(10)
        ax.set_box_aspect(0.72)
    handles, labels = unique_legend((ax_cnot, ax_t, ax_rep))
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=min(len(labels), 8),
        fontsize=8.5,
        frameon=False,
        handlelength=1.35,
        handletextpad=0.35,
        columnspacing=0.7,
        borderaxespad=0.0,
    )
    fig.tight_layout(rect=(0.0, 0.13, 1.0, 1.0), w_pad=0.55)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "hncc_resource_tradeoff_vs_eps.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


def make_time_tradeoff_plot(
    time_rows: Mapping[float, Sequence[ResourceRow]],
    fixed_eps: float,
    out_dir: Path,
) -> None:
    times = np.array(sorted(time_rows), dtype=float)
    gate_methods = sorted({row.method for rows in time_rows.values() for row in rows}, key=method_sort_key)
    rep_methods = sorted(
        {repetition_display_method(row.method) for rows in time_rows.values() for row in rows},
        key=method_sort_key,
    )

    fig, (ax_cnot, ax_t, ax_rep) = plt.subplots(1, 3, figsize=(10.8, 3.2))
    for method in gate_methods:
        cnot_values = []
        t_values = []
        for time in times:
            row = next(row for row in time_rows[float(time)] if row.method == method)
            cnot_values.append(row.max_cnot)
            t_values.append(max_t_gates(row))
        ax_cnot.loglog(
            times,
            cnot_values,
            marker_for_method(method),
            ms=3.4,
            label=method,
            **line_style_for_method(method),
        )
        ax_t.loglog(
            times,
            t_values,
            marker_for_method(method),
            ms=3.4,
            label=method,
            **line_style_for_method(method),
        )

    for method in rep_methods:
        values = []
        for time in times:
            row = next(
                row
                for row in time_rows[float(time)]
                if repetition_display_method(row.method) == method
            )
            values.append(row.repetitions * fixed_eps**2)
        ax_rep.semilogx(
            times,
            values,
            marker_for_method(method),
            ms=3.4,
            label=method,
            **line_style_for_method(method),
        )

    ax_cnot.set_ylabel(r"Max CNOT gates per run")
    ax_t.set_ylabel(r"Max $T$ gates per run")
    ax_rep.set_ylabel(r"Repetition overhead $N_{\rm rep}\epsilon^2$")
    for ax in (ax_cnot, ax_t, ax_rep):
        ax.set_xlabel(r"Simulation time $t$")
        ax.grid(True, which="both", ls=":", lw=0.55)
        ax.tick_params(axis="both", labelsize=8.5)
        ax.xaxis.label.set_size(10)
        ax.yaxis.label.set_size(10)
        ax.set_box_aspect(0.72)
    handles, labels = unique_legend((ax_cnot, ax_t, ax_rep))
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=min(len(labels), 8),
        fontsize=8.5,
        frameon=False,
        handlelength=1.35,
        handletextpad=0.35,
        columnspacing=0.7,
        borderaxespad=0.0,
    )
    fig.tight_layout(rect=(0.0, 0.13, 1.0, 1.0), w_pad=0.55)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "hncc_resource_tradeoff_vs_t.pdf"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


def method_sort_key(method: str) -> Tuple[int, str]:
    if method == "Trotter":
        return (0, method)
    if method.startswith("S"):
        return (0, method)
    if method.startswith("RE"):
        return (1, method)
    if method.startswith("NCC"):
        return (2, method)
    if method.startswith("HNCC"):
        return (3, method)
    return (4, method)


def repetition_display_method(method: str) -> str:
    if method.startswith("S"):
        return "Trotter"
    return method


def legend_label(n_key: int, method: str, n_count: int) -> str:
    return method


def marker_for_method(method: str) -> str:
    if method == "Trotter":
        return "--o"
    if method == "S2":
        return "--o"
    if method == "S4":
        return "-.s"
    if method == "S6":
        return ":^"
    if method.startswith("RE2"):
        return "--D"
    if method.startswith("RE4"):
        return "--P"
    if method == "NCC2":
        return "-.D"
    if method == "NCC4":
        return "-.P"
    if method == "HNCC2":
        return "-o"
    if method == "HNCC4":
        return "-s"
    return "-x"


def line_style_for_method(method: str) -> Dict[str, object]:
    colors = {
        "Trotter": "#5f6368",
        "S2": "#5f6368",
        "S4": "#8c8c8c",
        "S6": "#b0b0b0",
        "RE2-3": "#1f77b4",
        "RE4-3": "#17becf",
        "NCC2": "#7b3294",
        "NCC4": "#984ea3",
        "HNCC2": "#d62728",
        "HNCC4": "#ff7f0e",
    }
    style: Dict[str, object] = {
        "color": colors.get(method, "#333333"),
        "linewidth": 1.35,
        "alpha": 0.72,
    }
    if method.startswith("HNCC"):
        style.update(
            {
                "linewidth": 2.35,
                "alpha": 1.0,
                "zorder": 5,
                "markeredgecolor": "white",
                "markeredgewidth": 0.45,
            }
        )
    elif method.startswith("NCC"):
        style.update({"linewidth": 1.65, "alpha": 0.88, "zorder": 4})
    elif method.startswith("RE"):
        style.update({"linewidth": 1.45, "alpha": 0.82, "zorder": 3})
    return style


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
    parser.add_argument("--scan", choices=("eps", "time", "both"), default="both", help="Which scans to run.")
    parser.add_argument("--out-dir", type=Path, default=Path("figures"), help="Output figure directory.")
    parser.add_argument("--no-plots", action="store_true", help="Print tables without writing figures.")
    return parser.parse_args()


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

    extrap_formulas = [richardson_formula(base, (1, 2, 6)) for base in args.extrap_bases]
    for formula in extrap_formulas:
        print(
            f"{formula.name}: base S{formula.base_order}, multipliers={formula.multipliers}, "
            f"coeffs={[round(c, 8) for c in formula.coeffs]}, ||a||_1={formula.norm1:.6g}"
        )

    formula_orders = sorted(set(args.orders) | set(args.extrap_bases) | set(args.ncc_orders) | set(args.hncc_orders))
    results = [
        build_estimate(
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

    if args.scan in ("eps", "both"):
        all_rows: Dict[int, Sequence[ResourceRow]] = {}
        for result in results:
            rows = resource_rows_for_time(result, args.eps, args.time, args.orders, extrap_formulas, args.sampling_overhead_cap)
            all_rows[result.n] = rows
            print_rows(f"N={result.n}, eps scan at t={args.time:g}", rows)
        if not args.no_plots:
            make_eps_tradeoff_plot(all_rows, args.eps, args.out_dir)

    if args.scan in ("time", "both"):
        # Keep the time-scan plot uncluttered by using the first N value.
        result = results[0]
        time_rows: Dict[float, Sequence[ResourceRow]] = {}
        for time in args.time_values:
            rows = resource_rows_for_time(
                result,
                [args.fixed_eps],
                time,
                args.orders,
                extrap_formulas,
                args.sampling_overhead_cap,
                scan_value=float(time),
            )
            time_rows[float(time)] = rows
            print_rows(f"N={result.n}, time scan at eps={args.fixed_eps:.1e}, t={time:g}", rows)
        if not args.no_plots:
            make_time_tradeoff_plot(time_rows, args.fixed_eps, args.out_dir)


if __name__ == "__main__":
    main()
