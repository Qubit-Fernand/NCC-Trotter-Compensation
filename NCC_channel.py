"""
Action-on-rho prototype for the channel version of NCC.

This file avoids materializing dense Liouville superoperator matrices in the
main path. Instead, channel terms are represented as sparse linear combinations
of maps of the form

    rho -> P_left @ rho @ P_right

where ``P_left`` and ``P_right`` are Pauli strings. The leading orders use the
channel-pairing idea from Section 3.3 of the PDF, while higher orders are built
by composing sparse channel actions directly.

The implementation is still intended for small-N validation because the sparse
expansions can grow quickly with the truncation order.
"""

import argparse
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.linalg import expm
from tqdm import tqdm

from Pauli_Hamiltonian_BCH import (
    build_periodic_ab,
    cached_pauli_matrix_from_label,
    pauli_decomposition_stream,
    phi_term,
    tilde_F_term,
)


# Lookup table for single-qubit Pauli products with labels 0,1,2,3 = I,X,Y,Z.
# (2, 1): (-1j, 3) means Y X = -i Z
SINGLE_QUBIT_PAULI_MULTIPLICATION_TABLE = {
    (0, 0): (1.0 + 0.0j, 0),
    (0, 1): (1.0 + 0.0j, 1),
    (0, 2): (1.0 + 0.0j, 2),
    (0, 3): (1.0 + 0.0j, 3),
    (1, 0): (1.0 + 0.0j, 1),
    (1, 1): (1.0 + 0.0j, 0),
    (1, 2): (1.0j, 3),
    (1, 3): (-1.0j, 2),
    (2, 0): (1.0 + 0.0j, 2),
    (2, 1): (-1.0j, 3),
    (2, 2): (1.0 + 0.0j, 0),
    (2, 3): (1.0j, 1),
    (3, 0): (1.0 + 0.0j, 3),
    (3, 1): (1.0j, 2),
    (3, 2): (-1.0j, 1),
    (3, 3): (1.0 + 0.0j, 0),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Action-on-rho channel NCC prototype.")
    parser.add_argument("--N", type=int, default=3, help="number of spins")
    parser.add_argument("--J", type=float, default=1.0, help="interaction strength")
    parser.add_argument("--h", type=float, default=1.0, help="field strength")
    parser.add_argument("--T", type=float, default=1.0, help="total evolution time")
    parser.add_argument("--r", type=int, default=20, help="number of Trotter segments")
    parser.add_argument("--trials", type=int, default=1000, help="Monte Carlo trials")
    parser.add_argument("--seed", type=int, default=7, help="RNG seed")
    parser.add_argument("--epsilon", type=float, default=0.01, help="target precision")
    parser.add_argument(
        "--s0",
        type=int,
        default=0,
        help="max compensated order (0 => ceil(log(4/epsilon)))",
    )
    parser.add_argument(
        "--q0",
        type=int,
        default=0,
        help="max BCH order kept in tilde_F construction (0 => ceil(log(2N/epsilon)))",
    )
    parser.add_argument(
        "--max_dense_qubits",
        type=int,
        default=3,
        help="small-system guard for sparse channel validation",
    )
    parser.add_argument(
        "--save_trials_list",
        action="store_true",
        help="store per-trial output states in the output npz",
    )
    return parser.parse_args()


def zero_density_matrix(num_qubits: int) -> np.ndarray:
    """Return |0...0><0...0|."""
    dim = 2**num_qubits
    rho = np.zeros((dim, dim), dtype=complex)
    rho[0, 0] = 1.0
    return rho


def apply_unitary_channel(unitary: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Apply rho -> U rho U^dagger."""
    return unitary @ rho @ unitary.conj().T


def apply_ad_commutator(operator: np.ndarray, rho: np.ndarray) -> np.ndarray:
    """Apply rho -> operator rho - rho operator."""
    return operator @ rho - rho @ operator


def identity_label(num_qubits: int) -> tuple[int, ...]:
    return (0,) * num_qubits


@lru_cache(maxsize=None)
def multiply_pauli_labels(label_a: tuple[int, ...], label_b: tuple[int, ...]) -> tuple[complex, tuple[int, ...]]:
    """Multiply two Pauli-string labels and return phase, product-label."""
    if len(label_a) != len(label_b):
        raise ValueError("Pauli labels must have the same length")

    phase = 1.0 + 0.0j
    product = []
    for single_a, single_b in zip(label_a, label_b):
        single_phase, single_product = SINGLE_QUBIT_PAULI_MULTIPLICATION_TABLE[(single_a, single_b)]
        phase *= single_phase
        product.append(single_product)
    return phase, tuple(product)


def compress_channel_term(term: dict[tuple[tuple[int, ...], tuple[int, ...]], complex], tol=1e-12):
    """Drop numerically tiny coefficients from a sparse channel term."""
    return {key: value for key, value in term.items() if abs(value) > tol}


def add_channel_terms(
    accumulator: dict[tuple[tuple[int, ...], tuple[int, ...]], complex],
    term: dict[tuple[tuple[int, ...], tuple[int, ...]], complex],
    scale: complex = 1.0 + 0.0j,
):
    """Add a scaled sparse channel term into the accumulator."""
    for key, value in term.items():
        accumulator[key] = accumulator.get(key, 0.0 + 0.0j) + scale * value
    return accumulator


def channel_term_from_operator_pauli(coeffs, labels, num_qubits: int):
    """Build ad_O from the Pauli expansion O = sum coeff * P."""
    identity = identity_label(num_qubits)
    term: dict[tuple[tuple[int, ...], tuple[int, ...]], complex] = {}
    for coeff, label in zip(coeffs, labels):
        term[(label, identity)] = term.get((label, identity), 0.0 + 0.0j) + coeff
        term[(identity, label)] = term.get((identity, label), 0.0 + 0.0j) - coeff
    return compress_channel_term(term)


def compose_channel_terms(
    left_term: dict[tuple[tuple[int, ...], tuple[int, ...]], complex],
    right_term: dict[tuple[tuple[int, ...], tuple[int, ...]], complex],
    tol=1e-12,
):
    """Return left_term o right_term, i.e. apply right_term then left_term."""
    composed: dict[tuple[tuple[int, ...], tuple[int, ...]], complex] = {}
    for (left_l, left_r), left_coeff in left_term.items():
        for (right_l, right_r), right_coeff in right_term.items():
            left_phase, product_left = multiply_pauli_labels(left_l, right_l)
            right_phase, product_right = multiply_pauli_labels(right_r, left_r)
            coeff = left_coeff * right_coeff * left_phase * right_phase
            key = (product_left, product_right)
            composed[key] = composed.get(key, 0.0 + 0.0j) + coeff
    return compress_channel_term(composed, tol=tol)


def apply_channel_term(
    term: dict[tuple[tuple[int, ...], tuple[int, ...]], complex],
    rho: np.ndarray,
) -> np.ndarray:
    """Apply a sparse left/right Pauli channel term to rho."""
    out = np.zeros_like(rho)
    for (left_label, right_label), coeff in term.items():
        left = cached_pauli_matrix_from_label(left_label)
        right = cached_pauli_matrix_from_label(right_label)
        out += coeff * (left @ rho @ right)
    return out


def channel_term_to_arrays(term):
    """Convert a sparse channel term to deterministic arrays for sampling."""
    items = sorted(term.items())
    coeffs = np.array([coeff for _, coeff in items], dtype=complex)
    left_labels = [labels[0] for labels, _ in items]
    right_labels = [labels[1] for labels, _ in items]
    l1_norm = float(np.sum(np.abs(coeffs)))
    if l1_norm <= 0:
        raise ValueError("empty channel decomposition")
    return coeffs, left_labels, right_labels, l1_norm


def channel_term_max_abs_diff(term_a, term_b) -> float:
    """Compare two sparse channel terms by the largest coefficient mismatch."""
    all_keys = set(term_a) | set(term_b)
    if not all_keys:
        return 0.0
    return float(max(abs(term_a.get(key, 0.0 + 0.0j) - term_b.get(key, 0.0 + 0.0j)) for key in all_keys))


def iter_matrix_units(dim: int):
    """Yield the matrix-unit basis E_ij."""
    for row in range(dim):
        for col in range(dim):
            basis = np.zeros((dim, dim), dtype=complex)
            basis[row, col] = 1.0
            yield basis


def basis_action_distance(action_a, action_b, dim: int) -> float:
    """Maximum Frobenius error on the matrix-unit basis."""
    max_err = 0.0
    for basis in iter_matrix_units(dim):
        diff = action_a(basis) - action_b(basis)
        max_err = max(max_err, float(np.linalg.norm(diff, ord="fro")))
    return max_err


def build_sparse_tilde_F_terms(Phi_channel_terms, num_qubits: int, k_order: int, q0: int, s0: int):
    """Return sparse channel terms C_s with \tilde F_{K,s}(x) = C_s x^s."""

    def iter_compositions(total, parts, lower, upper):
        if parts == 1:
            if lower <= total <= upper:
                yield (total,)
            return
        min_rest = (parts - 1) * lower
        max_rest = (parts - 1) * upper
        start = max(lower, total - max_rest)
        stop = min(upper, total - min_rest)
        for first in range(start, stop + 1):
            for rest in iter_compositions(total - first, parts - 1, lower, upper):
                yield (first,) + rest

    id_label = identity_label(num_qubits)
    identity_term = {(id_label, id_label): 1.0 + 0.0j}
    tilde_F_terms = {}
    q_min = k_order + 1
    for s in range(q_min, s0 + 1):
        total_term: dict[tuple[tuple[int, ...], tuple[int, ...]], complex] = {}
        max_parts = s // q_min
        for j in range(1, max_parts + 1):
            for q_tuple in iter_compositions(s, j, q_min, q0):
                product_term = identity_term
                for q in q_tuple:
                    product_term = compose_channel_terms(product_term, Phi_channel_terms[q])
                add_channel_terms(total_term, product_term, scale=1.0 / math.factorial(j))
        tilde_F_terms[s] = compress_channel_term(total_term)
    return tilde_F_terms


def paired_channel_parameters(eta_pair_sum: float) -> dict:
    """Solve sin(theta) / cos(theta)^2 = eta_pair_sum and return pairing data."""
    if eta_pair_sum <= 0:
        return {
            "theta": 0.0,
            "sin_theta": 0.0,
            "cos_theta": 1.0,
            "cos_sq": 1.0,
            "pair_scale": 1.0,
            "unitary_branch_prob": 1.0,
        }

    sin_theta = (2.0 * eta_pair_sum) / (math.sqrt(1.0 + 4.0 * eta_pair_sum**2) + 1.0)
    theta = math.asin(min(1.0, max(-1.0, sin_theta)))
    cos_theta = math.cos(theta)
    cos_sq = cos_theta**2
    pair_scale = (1.0 + sin_theta**2) / cos_sq
    unitary_branch_prob = 1.0 / (1.0 + sin_theta**2)
    return {
        "theta": theta,
        "sin_theta": sin_theta,
        "cos_theta": cos_theta,
        "cos_sq": cos_sq,
        "pair_scale": pair_scale,
        "unitary_branch_prob": unitary_branch_prob,
    }


@lru_cache(maxsize=None)
def build_static_data(n, q0, s0, epsilon=0.01, j=1.0, h=1.0, K=1, max_dense_qubits=3):
    """Precompute r-independent operator data and sparse channel expansions."""
    if n > max_dense_qubits:
        raise ValueError(
            f"action-on-rho channel prototype only supports N <= {max_dense_qubits}; " f"received N={n}. For larger N, a locality-based sampler is still needed."
        )

    A_mat, B_mat = build_periodic_ab(n, j, h)
    dim = 2**n
    identity = np.eye(dim, dtype=complex)

    Phi_terms = phi_term(A_mat, B_mat, q0)
    tilde_F_operator_terms = tilde_F_term(Phi_terms, K, q0, s0)

    Phi_channel_terms = {}
    # each Phi_channel = sum_permutation [ad_X1, ad_X2, ...] = sum ad_[X_1, X_2, ...]
    # = sum [X_1, X_2, ...]\dot - \dot sum [X_1, X_2, ...] = Phi_operator \dot - \dot Phi_operator
    for order in range(K + 1, q0 + 1):
        coeffs, labels, _ = pauli_decomposition_stream(Phi_terms[order])
        Phi_channel_terms[order] = channel_term_from_operator_pauli(coeffs, labels, n)

    tilde_F_channel_terms = build_sparse_tilde_F_terms(Phi_channel_terms, n, K, q0, s0)

    F_terms = {}
    pairable_orders = []
    tail_orders = []
    pair_validation_errors = {}

    for order in range(K + 1, s0 + 1):
        if order <= 2 * K + 1:
            # verify that the leading-order term in operator F is anti-Hermitian.
            operator_term = tilde_F_operator_terms[order]
            antiherm_err = float(np.linalg.norm(operator_term + operator_term.conj().T, ord="fro"))
            if antiherm_err > 1e-8:
                raise ValueError(f"leading order s={order} is not anti-Hermitian enough for channel pairing " f"(antiherm_err={antiherm_err:.3e})")

            # leading-order F only contains single Phi, = ad_{Phi operator} = ad_{F operator}
            coeffs, labels, l1_norm = pauli_decomposition_stream(operator_term, antihermitian=True)
            direct_channel_term = channel_term_from_operator_pauli(coeffs, labels, n)
            pair_validation_errors[order] = channel_term_max_abs_diff(
                tilde_F_channel_terms[order],
                direct_channel_term,
            )
            pairable_orders.append(order)
            F_terms[order] = {
                "kind": "pair",
                "coeffs": coeffs,
                "labels": labels,
                "l1_norm": l1_norm,
                "antiherm_err": antiherm_err,
            }
        else:
            coeffs, left_labels, right_labels, l1_norm = channel_term_to_arrays(tilde_F_channel_terms[order])
            tail_orders.append(order)
            F_terms[order] = {
                "kind": "tail",
                "coeffs": coeffs,
                "left_labels": left_labels,
                "right_labels": right_labels,
                "l1_norm": l1_norm,
            }

    return {
        "A_mat": A_mat,
        "B_mat": B_mat,
        "h_total": A_mat + B_mat,
        "identity": identity,
        "dim": dim,
        "Phi_terms": Phi_terms,
        "Phi_channel_terms": Phi_channel_terms,
        "tilde_F_operator_terms": tilde_F_operator_terms,
        "tilde_F_channel_terms": tilde_F_channel_terms,
        "F_terms": F_terms,
        "s_orders": list(range(K + 1, s0 + 1)),
        "pairable_orders": pairable_orders,
        "tail_orders": tail_orders,
        "pair_validation_errors": pair_validation_errors,
        "K": K,
        "q0": q0,
        "s0": s0,
        "n": n,
        "epsilon": epsilon,
    }


def build_tilde_V(static, t_total, r, validation_tol=1e-10):
    """Build action-on-rho data for one step size."""
    A_mat = static["A_mat"]
    B_mat = static["B_mat"]
    h_total = static["h_total"]
    identity = static["identity"]
    dim = static["dim"]
    F_terms = static["F_terms"]
    s_orders = static["s_orders"]
    pairable_orders = static["pairable_orders"]
    tail_orders = static["tail_orders"]
    tilde_F_channel_terms = static["tilde_F_channel_terms"]

    t = t_total / r
    s1 = expm(-1j * B_mat * t) @ expm(-1j * A_mat * t)
    step_exact = expm(-1j * h_total * t)
    U_exact = expm(-1j * h_total * t_total)
    V_exact = step_exact @ expm(1j * A_mat * t) @ expm(1j * B_mat * t)

    eta = {order: F_terms[order]["l1_norm"] * (t**order) for order in s_orders}
    eta_pair_sum = float(sum(eta[order] for order in pairable_orders))
    pair_data = paired_channel_parameters(eta_pair_sum)
    pair_scale = pair_data["pair_scale"]

    raw_weights = {}
    if eta_pair_sum > 0:
        for order in pairable_orders:
            raw_weights[order] = pair_scale * eta[order] / eta_pair_sum
    for order in tail_orders:
        raw_weights[order] = eta[order]
    raw_total = float(sum(raw_weights.values()))
    p_order = np.array([raw_weights[order] / raw_total for order in s_orders], dtype=float)

    component_data = {}
    component_probs = {}
    for order in s_orders:
        data = F_terms[order]
        probs = np.abs(data["coeffs"]) / data["l1_norm"]
        components = []
        if data["kind"] == "pair":
            for prob, coeff, label in zip(probs, data["coeffs"], data["labels"]):
                phase = coeff / (1j * abs(coeff))
                W_mat = phase * cached_pauli_matrix_from_label(label)
                hermitian_err = float(np.linalg.norm(W_mat - W_mat.conj().T, ord="fro"))
                if hermitian_err > 1e-10:
                    raise ValueError(f"sampled W is not Hermitian (herm_err={hermitian_err:.3e})")
                paired_unitary = pair_data["cos_theta"] * identity + 1j * pair_data["sin_theta"] * W_mat
                components.append(
                    {
                        "prob": float(prob),
                        "W_mat": W_mat,
                        "paired_unitary": paired_unitary,
                    }
                )
        else:
            for prob, coeff, left_label, right_label in zip(
                probs,
                data["coeffs"],
                data["left_labels"],
                data["right_labels"],
            ):
                components.append(
                    {
                        "prob": float(prob),
                        "phase": coeff / abs(coeff),
                        "left_mat": cached_pauli_matrix_from_label(left_label),
                        "right_mat": cached_pauli_matrix_from_label(right_label),
                    }
                )
        component_data[order] = components
        component_probs[order] = np.array([component["prob"] for component in components], dtype=float)

    def apply_tilde_V_taylor(rho: np.ndarray) -> np.ndarray:
        out = rho.copy()
        for order in s_orders:
            out += (t**order) * apply_channel_term(tilde_F_channel_terms[order], rho)
        return out

    def apply_pair_component_expectation(component: dict, rho: np.ndarray) -> np.ndarray:
        return (rho + eta_pair_sum * apply_ad_commutator(1j * component["W_mat"], rho)) / pair_scale

    def apply_tilde_V_compensation(rho: np.ndarray) -> np.ndarray:
        out = np.zeros_like(rho)
        for order in s_orders:
            weight = raw_weights[order]
            data = F_terms[order]
            for component in component_data[order]:
                if data["kind"] == "pair":
                    out += weight * component["prob"] * apply_pair_component_expectation(component, rho)
                else:
                    out += weight * component["prob"] * component["phase"] * (component["left_mat"] @ rho @ component["right_mat"])
        return out

    def apply_uncompensated_single_step(rho: np.ndarray) -> np.ndarray:
        return apply_unitary_channel(s1, rho)

    def apply_exact_single_step(rho: np.ndarray) -> np.ndarray:
        return apply_unitary_channel(step_exact, rho)

    def apply_compensated_single_step(rho: np.ndarray) -> np.ndarray:
        return apply_tilde_V_compensation(apply_uncompensated_single_step(rho))

    def repeat_action(action, rho: np.ndarray, num_steps: int) -> np.ndarray:
        out = rho.copy()
        for _ in range(num_steps):
            out = action(out)
        return out

    def apply_uncompensated_total(rho: np.ndarray) -> np.ndarray:
        return repeat_action(apply_uncompensated_single_step, rho, r)

    def apply_compensated_total(rho: np.ndarray) -> np.ndarray:
        return repeat_action(apply_compensated_single_step, rho, r)

    def apply_exact_total(rho: np.ndarray) -> np.ndarray:
        return apply_unitary_channel(U_exact, rho)

    validation_error = basis_action_distance(apply_tilde_V_compensation, apply_tilde_V_taylor, dim)
    if validation_error > validation_tol:
        raise ValueError(f"channel tilde_V mismatch between compensation expectation and Taylor action: {validation_error:.3e}")

    deterministic_bias = basis_action_distance(apply_compensated_total, apply_exact_total, dim)
    uncompensated_total_error = basis_action_distance(apply_uncompensated_total, apply_exact_total, dim)
    single_step_error_before = basis_action_distance(apply_uncompensated_single_step, apply_exact_single_step, dim)
    single_step_expectation_bias = basis_action_distance(
        apply_compensated_single_step,
        apply_exact_single_step,
        dim,
    )

    return {
        "t": t,
        "s1": s1,
        "step_exact": step_exact,
        "U_exact": U_exact,
        "V_exact": V_exact,
        "eta": eta,
        "eta_pair_sum": eta_pair_sum,
        "raw_weights": raw_weights,
        "raw_total": raw_total,
        "p_order": p_order,
        "component_data": component_data,
        "component_probs": component_probs,
        "pair_orders": tuple(pairable_orders),
        "validation_error": validation_error,
        "deterministic_bias": deterministic_bias,
        "uncompensated_total_error": uncompensated_total_error,
        "single_step_error_before": single_step_error_before,
        "single_step_expectation_bias": single_step_expectation_bias,
        "apply_tilde_V_taylor": apply_tilde_V_taylor,
        "apply_tilde_V_compensation": apply_tilde_V_compensation,
        "apply_uncompensated_single_step": apply_uncompensated_single_step,
        "apply_compensated_single_step": apply_compensated_single_step,
        "apply_exact_single_step": apply_exact_single_step,
        "apply_uncompensated_total": apply_uncompensated_total,
        "apply_compensated_total": apply_compensated_total,
        "apply_exact_total": apply_exact_total,
        **pair_data,
    }


def sample_channel_then_compensate(
    rng: np.random.Generator,
    static: dict,
    evolution_data: dict,
    order: int,
    rho: np.ndarray,
):
    """Sample one compensated remainder channel and apply it directly to rho."""
    del static
    components = evolution_data["component_data"][order]
    probs = evolution_data["component_probs"][order]
    idx = int(rng.choice(len(components), p=probs))
    component = components[idx]

    if order in evolution_data.get("pair_orders", ()):
        if rng.random() < evolution_data["unitary_branch_prob"]:
            out = apply_unitary_channel(component["paired_unitary"], rho)
        else:
            out = -(component["W_mat"] @ rho @ component["W_mat"])
        return evolution_data["raw_total"] * out

    return evolution_data["raw_total"] * (component["phase"] * (component["left_mat"] @ rho @ component["right_mat"]))


def sample_compensated_single_step(
    rng: np.random.Generator,
    static: dict,
    evolution_data: dict,
    rho: np.ndarray,
) -> np.ndarray:
    """Sample one full compensated Trotter step and apply it to rho."""
    rho_after_trotter = apply_unitary_channel(evolution_data["s1"], rho)
    order = int(rng.choice(static["s_orders"], p=evolution_data["p_order"]))
    return sample_channel_then_compensate(rng, static, evolution_data, order, rho_after_trotter)


def main():
    args = parse_args()
    n = args.N
    j = args.J
    h = args.h
    t_total = args.T
    r = args.r
    trials = args.trials
    epsilon = args.epsilon
    K = 1

    s0 = int(np.ceil(np.log(4 / epsilon))) if args.s0 <= 0 else args.s0
    s0 = max(3, s0)
    q0 = int(np.ceil(np.log(2 * n / epsilon))) if args.q0 <= 0 else args.q0
    q0 = max(3, q0)

    static = build_static_data(
        n=n,
        q0=q0,
        s0=s0,
        epsilon=epsilon,
        j=j,
        h=h,
        K=K,
        max_dense_qubits=args.max_dense_qubits,
    )
    evolution_data = build_tilde_V(static, t_total, r)

    rho0 = zero_density_matrix(n)
    rho_single_exact = evolution_data["apply_exact_single_step"](rho0)
    rho_single_before = evolution_data["apply_uncompensated_single_step"](rho0)
    rho_single_deterministic = evolution_data["apply_compensated_single_step"](rho0)

    rho_total_exact = evolution_data["apply_exact_total"](rho0)
    rho_total_before = evolution_data["apply_uncompensated_total"](rho0)
    rho_total_deterministic = evolution_data["apply_compensated_total"](rho0)

    print("action-on-rho channel prototype:", True)
    print("N:", n, "q0:", q0, "s0:", s0, "r:", r)
    print("time step:", evolution_data["t"])
    print("pairable orders:", static["pairable_orders"])
    print("tail orders:", static["tail_orders"])
    print("pair validation errors:", static["pair_validation_errors"])
    print("eta_pair_sum:", evolution_data["eta_pair_sum"])
    print("eta by order:", {s: evolution_data["eta"][s] for s in static["s_orders"]})
    print("pair theta:", evolution_data["theta"])
    print("pair scale:", evolution_data["pair_scale"])
    print("raw weights:", evolution_data["raw_weights"])
    print("tilde_V compensation-vs-Taylor check:", evolution_data["validation_error"])
    print("single-step basis error before:", evolution_data["single_step_error_before"])
    print("single-step basis expectation bias:", evolution_data["single_step_expectation_bias"])
    print("total basis error before:", evolution_data["uncompensated_total_error"])
    print("total basis expectation bias:", evolution_data["deterministic_bias"])

    rng = np.random.default_rng(seed=args.seed)

    def single_step_channel_sampling(num_trials):
        rho_list = [] if args.save_trials_list else None
        average = np.zeros_like(rho0)
        for _ in tqdm(range(num_trials), desc="single-step channel trials"):
            sample = sample_compensated_single_step(rng, static, evolution_data, rho0)
            average += sample
            if rho_list is not None:
                rho_list.append(sample)
        average /= num_trials
        return rho_list, average

    rho_single_list, rho_single_avg = single_step_channel_sampling(trials)
    single_step_sample_error = float(np.linalg.norm(rho_single_avg - rho_single_exact, ord="fro"))
    single_step_fluctuation = float(np.linalg.norm(rho_single_avg - rho_single_deterministic, ord="fro"))
    single_step_error_before_state = float(np.linalg.norm(rho_single_before - rho_single_exact, ord="fro"))
    single_step_expectation_bias_state = float(np.linalg.norm(rho_single_deterministic - rho_single_exact, ord="fro"))

    print("single-step state error before:", single_step_error_before_state)
    print("single-step state sample error after compensation:", single_step_sample_error)
    print("single-step state sample fluctuation:", single_step_fluctuation)
    print("single-step state expectation bias:", single_step_expectation_bias_state)

    def multi_step_channel_sampling(num_trials):
        rho_list = [] if args.save_trials_list else None
        average = np.zeros_like(rho0)
        for _ in tqdm(range(num_trials), desc="multi-step channel trials"):
            rho = rho0.copy()
            for _ in range(r):
                rho = sample_compensated_single_step(rng, static, evolution_data, rho)
            average += rho
            if rho_list is not None:
                rho_list.append(rho)
        average /= num_trials
        return rho_list, average

    rho_total_list, rho_total_avg = multi_step_channel_sampling(trials)
    total_sample_error = float(np.linalg.norm(rho_total_avg - rho_total_exact, ord="fro"))
    total_sample_fluctuation = float(np.linalg.norm(rho_total_avg - rho_total_deterministic, ord="fro"))
    total_error_before_state = float(np.linalg.norm(rho_total_before - rho_total_exact, ord="fro"))
    total_expectation_bias_state = float(np.linalg.norm(rho_total_deterministic - rho_total_exact, ord="fro"))

    print("total state error before:", total_error_before_state)
    print("multi-step state sample error after compensation:", total_sample_error)
    print("multi-step state sample fluctuation:", total_sample_fluctuation)
    print("multi-step state expectation bias:", total_expectation_bias_state)

    data_dir = Path("data/no_search")
    data_dir.mkdir(parents=True, exist_ok=True)
    out = data_dir / f"NCC_channel_trials{trials}_N{args.N}_r{args.r}_q{q0}_s{s0}.npz"
    output_payload = dict(
        rho0=rho0,
        rho_single_exact=rho_single_exact,
        rho_single_before=rho_single_before,
        rho_single_deterministic=rho_single_deterministic,
        rho_single_average=rho_single_avg,
        rho_total_exact=rho_total_exact,
        rho_total_before=rho_total_before,
        rho_total_deterministic=rho_total_deterministic,
        rho_total_average=rho_total_avg,
        single_step_basis_error_before=evolution_data["single_step_error_before"],
        single_step_basis_expectation_bias=evolution_data["single_step_expectation_bias"],
        total_basis_error_before=evolution_data["uncompensated_total_error"],
        total_basis_expectation_bias=evolution_data["deterministic_bias"],
        validation_error=evolution_data["validation_error"],
        single_step_state_error_before=single_step_error_before_state,
        single_step_state_sample_error=single_step_sample_error,
        single_step_state_fluctuation=single_step_fluctuation,
        single_step_state_expectation_bias=single_step_expectation_bias_state,
        total_state_error_before=total_error_before_state,
        total_state_sample_error=total_sample_error,
        total_state_fluctuation=total_sample_fluctuation,
        total_state_expectation_bias=total_expectation_bias_state,
        N=n,
        J=j,
        h=h,
        T=t_total,
        r=r,
        trials=trials,
        seed=args.seed,
        epsilon=epsilon,
        q0=q0,
        s0=s0,
        eta_pair_sum=evolution_data["eta_pair_sum"],
        pair_scale=evolution_data["pair_scale"],
        raw_total=evolution_data["raw_total"],
        theta=evolution_data["theta"],
        sin_theta=evolution_data["sin_theta"],
        cos_sq=evolution_data["cos_sq"],
        save_trials_list=args.save_trials_list,
    )
    if args.save_trials_list:
        output_payload["rho_single_list"] = np.stack(rho_single_list)
        output_payload["rho_total_list"] = np.stack(rho_total_list)
    np.savez(out, **output_payload)
    print("saving results to:", out)


if __name__ == "__main__":
    main()
