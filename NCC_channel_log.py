"""
No rejection where x absorbed in Phi, Phi only contains matrix as coeff.
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
        "--save_trials_list",
        action="store_true",
        help="store per-trial output states in the output npz",
    )
    return parser.parse_args()


def computational_basis_density_matrices(num_qubits: int) -> tuple[np.ndarray, list[str]]:
    """Return all computational-basis pure states as density matrices."""
    dim = 2**num_qubits
    basis_states = np.zeros((dim, dim, dim), dtype=complex)
    labels = []
    for idx in range(dim):
        basis_states[idx, idx, idx] = 1.0
        labels.append(format(idx, f"0{num_qubits}b"))
    return basis_states, labels


def apply_unitary_channel(unitary: np.ndarray, rho: np.ndarray, sign: complex = 1.0 + 0.0j) -> np.ndarray:
    """Apply rho -> sign * U rho U^dagger."""
    return sign * (unitary @ rho @ unitary.conj().T)


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


def trace_norm(matrix: np.ndarray) -> float:
    """Return the trace norm (Schatten-1 norm) of a matrix."""
    return float(np.sum(np.linalg.svd(matrix, compute_uv=False)))


def apply_signed_unitary_channel_to_basis_states(sign: complex, unitary: np.ndarray) -> np.ndarray:
    """Apply a signed unitary channel to all computational-basis pure states at once."""
    columns = unitary.T.copy()
    return sign * np.einsum("ai,bi->iab", columns, columns.conj(), optimize=True)


def max_trace_norm_error(states_a: np.ndarray, states_b: np.ndarray) -> tuple[float, int, np.ndarray]:
    """Return the maximum trace-norm error over two state batches."""
    errors = np.array([trace_norm(a - b) for a, b in zip(states_a, states_b)], dtype=float)
    worst_index = int(np.argmax(errors))
    return float(errors[worst_index]), worst_index, errors


# used for F_s into Phi_q_tuple decompostion.
def iter_compositions(total: int, parts: int, lower: int, upper: int):
    """Yield ordered compositions with bounded part sizes."""
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


# channel_F_tail = ad_phi * ad_phi ...., decompose each ad_phi into ad_Pauli
# when sampling, each ad_Pauli is sampled as either +e^{+i pi/4 ad_P} or -e^{-i pi/4 ad_P}
def build_phi_layer_in_tail_F(Phi_terms: dict[int, np.ndarray], n: int, k_order: int, q0: int):
    """Store only operator Phi_q and its anti-Hermitian Pauli decomposition."""
    identity = np.eye(2**n, dtype=complex)
    phi_layer_in_tail_F = {}
    for order in range(k_order + 1, q0 + 1):
        antiherm_err = float(np.linalg.norm(Phi_terms[order] + Phi_terms[order].conj().T, ord="fro"))
        if antiherm_err > 1e-8:
            raise ValueError(f"Phi_{order} is not anti-Hermitian enough for channel sampling (antiherm_err={antiherm_err:.3e})")
        coeffs, labels, l1_norm = pauli_decomposition_stream(Phi_terms[order], antihermitian=True)
        probs = np.abs(coeffs) / l1_norm
        Paulis = []
        for prob, coeff, label in zip(probs, coeffs, labels):
            pauli_sign = coeff / (1j * abs(coeff))
            if abs(np.imag(pauli_sign)) > 1e-10 or not np.isclose(abs(np.real(pauli_sign)), 1.0, atol=1e-10):
                raise ValueError(f"Phi_{order} Pauli coefficient has unexpected phase {pauli_sign}")
            pauli_sign = float(np.real(pauli_sign))
            pauli_mat = cached_pauli_matrix_from_label(label)
            Paulis.append(
                {
                    "prob": float(prob),
                    "pauli_sign": pauli_sign,
                    "pauli_mat": pauli_mat,
                    # Phi_q is anti-Hermitian, so each sampled term is i * (± P).
                    # We keep the base Hermitian Pauli P and absorb the sign into
                    # which exp(± i pi/4 P) branch is called "plus" or "minus".
                    "plus_unitary": (identity + 1j * pauli_sign * pauli_mat) / math.sqrt(2.0),
                    "minus_unitary": (identity - 1j * pauli_sign * pauli_mat) / math.sqrt(2.0),
                }
            )
        phi_layer_in_tail_F[order] = {
            "coeffs": coeffs,
            "labels": labels,
            "l1_norm": l1_norm,
            "antiherm_err": antiherm_err,
            # One ad_{iP} layer is sampled as either +exp(+i pi/4 ad_P)
            # or -exp(-i pi/4 ad_P), so the executable sampling 1-norm
            # for this Phi_q layer picks up the expected split factor 2.
            "layer_sampling_l1_norm": 2.0 * l1_norm,
            "Paulis": Paulis,
            "Pauli_probs": np.array([Pauli["prob"] for Pauli in Paulis], dtype=float),
        }
    return phi_layer_in_tail_F


def tail_sampling_ad_phi_tuples(phi_layer_in_tail_F: dict[int, dict], k_order: int, q0: int, s0: int):
    """
    Build tail phi_tuples without explicitly materializing F_channel.

    For tail orders we only store how F_{K,s} is assembled as products of
    ad_{Phi_q}. When sampling, each ad_{Phi_q} is expanded into Pauli terms and
    each ad_{iP} is sampled as either +e^{+i pi/4 ad_P} or -e^{-i pi/4 ad_P}.
    """
    tail_phi_tuple_terms = {}
    q_min = k_order + 1
    for order in range(2 * k_order + 2, s0 + 1):
        phi_tuples = []
        max_parts = order // q_min
        for num_parts in range(1, max_parts + 1):
            # Tail F_s is stored as phi_tuples with j = num_parts layers of ad_{Phi_q}.
            # The combinatorial coefficient is therefore 1/j!, not 1/s!.
            factorial_j_scale = 1.0 / math.factorial(num_parts)
            for q_tuple in iter_compositions(order, num_parts, q_min, q0):
                layer_l1_norm_product = float(np.prod([phi_layer_in_tail_F[q]["l1_norm"] for q in q_tuple], dtype=float))
                layer_sampling_l1_product = float(np.prod([phi_layer_in_tail_F[q]["layer_sampling_l1_norm"] for q in q_tuple], dtype=float))
                phi_tuples.append(
                    {
                        "q_tuple": q_tuple,
                        "factorial_j_scale": factorial_j_scale,
                        "layer_l1_norm_product": layer_l1_norm_product,
                        "layer_sampling_l1_product": layer_sampling_l1_product,
                        "eta_weight": factorial_j_scale * layer_l1_norm_product,
                        "sampling_weight": factorial_j_scale * layer_sampling_l1_product,
                    }
                )
        if not phi_tuples:
            raise ValueError(f"tail order s={order} has no valid Phi-composition phi_tuples")
        eta_l1_norm = float(sum(phi_tuple["eta_weight"] for phi_tuple in phi_tuples))
        sampling_l1_norm = float(sum(phi_tuple["sampling_weight"] for phi_tuple in phi_tuples))
        phi_tuple_probs = np.array([phi_tuple["sampling_weight"] / sampling_l1_norm for phi_tuple in phi_tuples], dtype=float)
        tail_phi_tuple_terms[order] = {
            "phi_tuples": phi_tuples,
            "eta_l1_norm": eta_l1_norm,
            "sampling_l1_norm": sampling_l1_norm,
            "phi_tuple_probs": phi_tuple_probs,
        }
    return tail_phi_tuple_terms


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
    # Section 5.2 normalization:
    # I +/- i eta ad_P = pair_scale * ((1-p) U_theta(.)U_theta^\dagger + p(-P.P))
    # with pair_scale = (1 + sin^2 theta) / cos^2 theta and p = sin^2 theta / (1 + sin^2 theta).
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
def build_static_data(n, q0, s0, j=1.0, h=1.0, K=1):
    """Precompute r-independent operator data and sampling phi_tuples."""
    A_mat, B_mat = build_periodic_ab(n, j, h)
    dim = 2**n
    identity = np.eye(dim, dtype=complex)

    Phi_terms = phi_term(A_mat, B_mat, q0)
    tilde_F_operator_terms = tilde_F_term(Phi_terms, K, q0, s0)
    phi_layer_in_tail_F = build_phi_layer_in_tail_F(Phi_terms, n, K, q0)
    tail_phi_tuple_terms = tail_sampling_ad_phi_tuples(phi_layer_in_tail_F, K, q0, s0)

    F_terms = {}
    pairable_orders = []
    tail_orders = []
    pair_validation_errors = {}

    for order in range(K + 1, s0 + 1):
        if order <= 2 * K + 1:
            tilde_operator_F = tilde_F_operator_terms[order]
            antiherm_err = float(np.linalg.norm(tilde_operator_F + tilde_operator_F.conj().T, ord="fro"))
            if antiherm_err > 1e-8:
                raise ValueError(f"leading order s={order} is not anti-Hermitian enough for channel pairing " f"(antiherm_err={antiherm_err:.3e})")

            coeffs, labels, l1_norm = pauli_decomposition_stream(tilde_operator_F, antihermitian=True)
            pair_validation_errors[order] = float(np.linalg.norm(tilde_operator_F - Phi_terms[order], ord="fro"))
            pairable_orders.append(order)
            F_terms[order] = {
                "kind": "pair",
                "coeffs": coeffs,
                "labels": labels,
                "l1_norm": l1_norm,
                "antiherm_err": antiherm_err,
            }
        else:
            tail_orders.append(order)
            F_terms[order] = {
                "kind": "tail",
                **tail_phi_tuple_terms[order],
            }

    return {
        "A_mat": A_mat,
        "B_mat": B_mat,
        "h_total": A_mat + B_mat,
        "identity": identity,
        "dim": dim,
        "Phi_terms": Phi_terms,
        "phi_layer_in_tail_F": phi_layer_in_tail_F,
        "tilde_F_operator_terms": tilde_F_operator_terms,
        "F_terms": F_terms,
        "s_orders": list(range(K + 1, s0 + 1)),
        "pairable_orders": pairable_orders,
        "tail_orders": tail_orders,
        "pair_validation_errors": pair_validation_errors,
        "K": K,
        "q0": q0,
        "s0": s0,
        "n": n,
    }


def build_tilde_V(static, t_total, r, validation_tol=1e-10):
    """Build action-on-rho data for one step size."""
    # step t/r is absorbed into the per-order coefficients below.
    A_mat = static["A_mat"]
    B_mat = static["B_mat"]
    h_total = static["h_total"]
    identity = static["identity"]
    dim = static["dim"]
    F_terms = static["F_terms"]
    s_orders = static["s_orders"]
    pairable_orders = static["pairable_orders"]
    tail_orders = static["tail_orders"]
    phi_layer_in_tail_F = static["phi_layer_in_tail_F"]

    t = t_total / r
    S1 = expm(-1j * B_mat * t) @ expm(-1j * A_mat * t)
    step_exact = expm(-1j * h_total * t)
    U_exact = expm(-1j * h_total * t_total)
    V_exact = step_exact @ expm(1j * A_mat * t) @ expm(1j * B_mat * t)

    eta = {}
    for order in s_orders:
        if F_terms[order]["kind"] == "pair":
            eta[order] = F_terms[order]["l1_norm"] * (t**order)
        else:
            # For tail orders, eta keeps the Pauli-1-norm semantics only:
            # eta_s = (sum over phi-tuples of 1/j! * prod_q ||Phi_q||_1) * t^s.
            eta[order] = F_terms[order]["eta_l1_norm"] * (t**order)
            # The executable sampler carries extra split factors from each ad_{iP} layer,
            # so keep its outer coefficient under a separate name.

    eta_pair_sum = float(sum(eta[order] for order in pairable_orders))
    pair_params = paired_channel_parameters(eta_pair_sum)
    pair_scale = pair_params["pair_scale"]

    raw_weights = {}
    if eta_pair_sum > 0:
        for order in pairable_orders:
            # Outer order sampling for leading terms follows the note:
            # weight_s = pair_scale * eta_s / sum_{pair orders} eta_s.
            raw_weights[order] = pair_scale * eta[order] / eta_pair_sum
    for order in tail_orders:
        # 2^j compared to eta because each ad_{Phi_q} layer is sampled as either +exp(+i pi/4 ad_P) or -exp(-i pi/4 ad_P).
        # Tail orders must still carry the step-size scaling t^s.
        # Compared with eta_s, sampling_l1_norm includes the extra 2^j split
        # from sampling each ad_{iP} layer by two unitary branches.
        raw_weights[order] = F_terms[order]["sampling_l1_norm"] * (t**order)
    raw_l1_norm_total = float(sum(raw_weights.values()))
    p_order = np.array([raw_weights[order] / raw_l1_norm_total for order in s_orders], dtype=float)

    pair_components = {}
    for order in s_orders:
        data = F_terms[order]
        if data["kind"] == "pair":
            Paulis = []
            probs = np.abs(data["coeffs"]) / data["l1_norm"]
            for prob, coeff, label in zip(probs, data["coeffs"], data["labels"]):
                phase = coeff / (1j * abs(coeff))
                W_mat = phase * cached_pauli_matrix_from_label(label)
                hermitian_err = float(np.linalg.norm(W_mat - W_mat.conj().T, ord="fro"))
                if hermitian_err > 1e-10:
                    raise ValueError(f"sampled W is not Hermitian (herm_err={hermitian_err:.3e})")
                # e^{i theta Pauli} = cos(theta) I + i sin(theta) Pauli
                rotation_branch_in_pair = pair_params["cos_theta"] * identity + 1j * pair_params["sin_theta"] * W_mat
                Paulis.append(
                    {
                        "prob": float(prob),
                        "W_mat": W_mat,
                        "rotation_branch_in_pair": rotation_branch_in_pair,
                    }
                )
            pair_components[order] = Paulis

    # F_s(rho) = sum 1/j! * ad_phi_q1 ad_phi_q2(rho)
    def apply_tail_F_channel(order: int, rho: np.ndarray) -> np.ndarray:
        out = np.zeros_like(rho)
        for phi_tuple in F_terms[order]["phi_tuples"]:
            tuple_out = rho
            for phi_order in reversed(phi_tuple["q_tuple"]):
                tuple_out = apply_ad_commutator(static["Phi_terms"][phi_order], tuple_out)
            out += phi_tuple["factorial_j_scale"] * tuple_out
        return out

    # tilde_V(rho) = sum F_s(rho)
    def apply_tilde_V_taylor(rho: np.ndarray) -> np.ndarray:
        out = rho.copy()
        for order in s_orders:
            if F_terms[order]["kind"] == "pair":
                out += (t**order) * apply_ad_commutator(static["tilde_F_operator_terms"][order], rho)
            else:
                out += (t**order) * apply_tail_F_channel(order, rho)
        return out

    def apply_tilde_V_expectation(rho: np.ndarray) -> np.ndarray:
        out = np.zeros_like(rho)
        for order in s_orders:
            weight = raw_weights[order]
            data = F_terms[order]
            if data["kind"] == "pair":
                for Pauli in pair_components[order]:
                    # for this part, weight contains eta_2/eta * pair_scale
                    # eta_2/eta * pair_scale (I + i eta ad_W)/pair_sacle
                    out += weight * Pauli["prob"] * (rho + eta_pair_sum * apply_ad_commutator(1j * Pauli["W_mat"], rho)) / pair_scale
            else:
                # for this part, weight contains sum 2^j eta_s = data["sampling_l1_norm"]
                out += weight * apply_tail_F_channel(order, rho) / data["sampling_l1_norm"]
        return out

    validation_states, _ = computational_basis_density_matrices(static["n"])
    validation_error = max(trace_norm(apply_tilde_V_expectation(rho) - apply_tilde_V_taylor(rho)) for rho in validation_states)
    if validation_error > validation_tol:
        raise ValueError(f"channel tilde_V mismatch between compensation expectation and Taylor action: {validation_error:.3e}")

    return {
        "t": t,
        "S1": S1,
        "step_exact": step_exact,
        "U_exact": U_exact,
        "V_exact": V_exact,
        "eta": eta,
        "eta_pair_sum": eta_pair_sum,
        "raw_weights": raw_weights,
        "raw_l1_norm_total": raw_l1_norm_total,
        "p_order": p_order,
        "pair_data": pair_components,
        "pair_orders": tuple(pairable_orders),
        "validation_error": validation_error,
        "apply_tilde_V_taylor": apply_tilde_V_taylor,
        "apply_tilde_V_expectation": apply_tilde_V_expectation,
        **pair_params,
    }

def sample_channel_then_compensate_descriptor(
    rng: np.random.Generator,
    static: dict,
    evolution_data: dict,
    order: int,
) -> tuple[complex, np.ndarray]:
    """
    Sample one compensated remainder channel as a signed unitary descriptor.

    The returned descriptor represents the action

        rho -> sign * U rho U^\dagger
    """
    if order in evolution_data.get("pair_orders", ()):
        Paulis = evolution_data["pair_data"][order]
        probs = np.array([Pauli["prob"] for Pauli in Paulis], dtype=float)
        idx = int(rng.choice(len(Paulis), p=probs))
        Pauli = Paulis[idx]
        if rng.random() < evolution_data["unitary_branch_prob"]:
            return evolution_data["raw_l1_norm_total"], Pauli["rotation_branch_in_pair"]
        return -evolution_data["raw_l1_norm_total"], Pauli["W_mat"]

    tail_data = static["F_terms"][order]
    phi_tuple_idx = int(rng.choice(len(tail_data["phi_tuples"]), p=tail_data["phi_tuple_probs"]))
    phi_tuple = tail_data["phi_tuples"][phi_tuple_idx]
    sign = 1.0 + 0.0j
    dim = static["dim"]
    unitary = np.eye(dim, dtype=complex)
    for phi_order in reversed(phi_tuple["q_tuple"]):
        phi_data = static["phi_layer_in_tail_F"][phi_order]
        Pauli_idx = int(rng.choice(len(phi_data["Paulis"]), p=phi_data["Pauli_probs"]))
        Pauli = phi_data["Paulis"][Pauli_idx]
        if rng.random() < 0.5:
            sampled_sign = 1.0 + 0.0j
            sampled_unitary = Pauli["plus_unitary"]
        else:
            sampled_sign = -1.0 + 0.0j
            sampled_unitary = Pauli["minus_unitary"]
        sign *= sampled_sign
        unitary = sampled_unitary @ unitary
    return evolution_data["raw_l1_norm_total"] * sign, unitary


def sample_trotter_step_descriptor(
    rng: np.random.Generator,
    static: dict,
    evolution_data: dict,
) -> tuple[complex, np.ndarray]:
    """Sample one full compensated Trotter step as a signed unitary descriptor."""
    order = int(rng.choice(static["s_orders"], p=evolution_data["p_order"]))
    sign, unitary = sample_channel_then_compensate_descriptor(rng, static, evolution_data, order)
    return sign, unitary @ evolution_data["S1"]


def sample_trajectory_descriptor(
    rng: np.random.Generator,
    static: dict,
    evolution_data: dict,
    num_steps: int,
) -> tuple[complex, np.ndarray]:
    """Sample a multi-step trajectory and compress it into one signed unitary descriptor."""
    sign = 1.0 + 0.0j
    unitary = np.eye(static["dim"], dtype=complex)
    for _ in range(num_steps):
        step_sign, step_unitary = sample_trotter_step_descriptor(rng, static, evolution_data)
        sign *= step_sign
        unitary = step_unitary @ unitary
    return sign, unitary


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
        j=j,
        h=h,
        K=K,
    )
    evolution_data = build_tilde_V(static, t_total, r)

    basis_states, basis_labels = computational_basis_density_matrices(n)
    rho_single_exact = np.array(
        [apply_unitary_channel(evolution_data["step_exact"], rho) for rho in basis_states],
        dtype=complex,
    )
    rho_single_before = np.array(
        [apply_unitary_channel(evolution_data["S1"], rho) for rho in basis_states],
        dtype=complex,
    )
    rho_single_deterministic = np.array(
        [evolution_data["apply_tilde_V_expectation"](rho) for rho in rho_single_before],
        dtype=complex,
    )

    rho_total_exact = np.array(
        [apply_unitary_channel(evolution_data["U_exact"], rho) for rho in basis_states],
        dtype=complex,
    )
    rho_total_before = basis_states.copy()
    rho_total_deterministic = basis_states.copy()
    for _ in range(r):
        rho_total_before = np.array(
            [apply_unitary_channel(evolution_data["S1"], rho) for rho in rho_total_before],
            dtype=complex,
        )
        rho_total_deterministic = np.array(
            [
                evolution_data["apply_tilde_V_expectation"](apply_unitary_channel(evolution_data["S1"], rho))
                for rho in rho_total_deterministic
            ],
            dtype=complex,
        )

    print("action-on-rho channel prototype:", True)
    print("N:", n, "q0:", q0, "s0:", s0, "r:", r)
    print("time step:", evolution_data["t"])
    print("pairable orders:", static["pairable_orders"])
    print("tail orders:", static["tail_orders"])
    print("pair validation errors:", static["pair_validation_errors"])
    print("eta_pair_sum:", evolution_data["eta_pair_sum"])
    print("eta by order:", {s: evolution_data["eta"][s] for s in static["s_orders"]})
    print("pair raw weights:", {s: evolution_data["raw_weights"][s] for s in static["pairable_orders"]})
    print("tail raw weights:", {s: evolution_data["raw_weights"][s] for s in static["tail_orders"]})
    print("pair theta:", evolution_data["theta"])
    print("pair scale:", evolution_data["pair_scale"])
    print("debug channel validation (all computational-basis states):", evolution_data["validation_error"])

    rng = np.random.default_rng(seed=args.seed)

    def single_step_channel_sampling(num_trials):
        rho_list = [] if args.save_trials_list else None
        average = np.zeros_like(rho_single_exact)
        for _ in tqdm(range(num_trials), desc="single-step channel trials"):
            sign, unitary = sample_trotter_step_descriptor(rng, static, evolution_data)
            sample = apply_signed_unitary_channel_to_basis_states(sign, unitary)
            average += sample
            if rho_list is not None:
                rho_list.append(sample)
        average /= num_trials
        return rho_list, average

    rho_single_list, rho_single_avg = single_step_channel_sampling(trials)
    single_step_sample_error, single_step_sample_argmax, single_step_sample_errors = max_trace_norm_error(rho_single_avg, rho_single_exact)
    single_step_fluctuation, single_step_fluctuation_argmax, single_step_fluctuations = max_trace_norm_error(rho_single_avg, rho_single_deterministic)
    single_step_error_before_state, single_step_before_argmax, single_step_before_errors = max_trace_norm_error(rho_single_before, rho_single_exact)
    single_step_expectation_bias_state, single_step_bias_argmax, single_step_bias_errors = max_trace_norm_error(rho_single_deterministic, rho_single_exact)

    print("single-step max trace distance before:", 0.5 * single_step_error_before_state, "at", basis_labels[single_step_before_argmax])
    print("single-step max trace distance expectation bias:", 0.5 * single_step_expectation_bias_state, "at", basis_labels[single_step_bias_argmax])
    print("single-step max trace distance sample fluctuation:", 0.5 * single_step_fluctuation, "at", basis_labels[single_step_fluctuation_argmax])
    print("single-step max trace distance after compensation:", 0.5 * single_step_sample_error, "at", basis_labels[single_step_sample_argmax])

    def multi_step_channel_sampling(num_trials):
        rho_list = [] if args.save_trials_list else None
        average = np.zeros_like(rho_total_exact)
        for _ in tqdm(range(num_trials), desc="multi-step channel trials"):
            sign, unitary = sample_trajectory_descriptor(rng, static, evolution_data, r)
            rho = apply_signed_unitary_channel_to_basis_states(sign, unitary)
            average += rho
            if rho_list is not None:
                rho_list.append(rho)
        average /= num_trials
        return rho_list, average

    rho_total_list, rho_total_avg = multi_step_channel_sampling(trials)
    total_sample_error, total_sample_argmax, total_sample_errors = max_trace_norm_error(rho_total_avg, rho_total_exact)
    total_sample_fluctuation, total_fluctuation_argmax, total_sample_fluctuations = max_trace_norm_error(rho_total_avg, rho_total_deterministic)
    total_error_before_state, total_before_argmax, total_before_errors = max_trace_norm_error(rho_total_before, rho_total_exact)
    total_expectation_bias_state, total_bias_argmax, total_bias_errors = max_trace_norm_error(rho_total_deterministic, rho_total_exact)

    print("total max trace distance before:", 0.5 * total_error_before_state, "at", basis_labels[total_before_argmax])
    print("total max trace distance expectation bias:", 0.5 * total_expectation_bias_state, "at", basis_labels[total_bias_argmax])
    print("total max trace distance sample fluctuation:", 0.5 * total_sample_fluctuation, "at", basis_labels[total_fluctuation_argmax])
    print("total max trace distance after compensation:", 0.5 * total_sample_error, "at", basis_labels[total_sample_argmax])
    data_dir = Path("data/no_search")
    data_dir.mkdir(parents=True, exist_ok=True)
    out = data_dir / f"NCC_channel_log_trials{trials}_N{args.N}_r{args.r}_q{q0}_s{s0}.npz"
    output_payload = dict(
        basis_labels=np.array(basis_labels),
        validation_error=evolution_data["validation_error"],
        single_step_state_error_before=single_step_error_before_state,
        single_step_state_sample_error=single_step_sample_error,
        single_step_state_fluctuation=single_step_fluctuation,
        single_step_state_expectation_bias=single_step_expectation_bias_state,
        single_step_state_error_before_state_index=single_step_before_argmax,
        single_step_state_sample_error_state_index=single_step_sample_argmax,
        single_step_state_fluctuation_state_index=single_step_fluctuation_argmax,
        single_step_state_expectation_bias_state_index=single_step_bias_argmax,
        single_step_state_error_before_all=single_step_before_errors,
        single_step_state_sample_error_all=single_step_sample_errors,
        single_step_state_fluctuation_all=single_step_fluctuations,
        single_step_state_expectation_bias_all=single_step_bias_errors,
        total_state_error_before=total_error_before_state,
        total_state_sample_error=total_sample_error,
        total_state_fluctuation=total_sample_fluctuation,
        total_state_expectation_bias=total_expectation_bias_state,
        total_state_error_before_state_index=total_before_argmax,
        total_state_sample_error_state_index=total_sample_argmax,
        total_state_fluctuation_state_index=total_fluctuation_argmax,
        total_state_expectation_bias_state_index=total_bias_argmax,
        total_state_error_before_all=total_before_errors,
        total_state_sample_error_all=total_sample_errors,
        total_state_fluctuation_all=total_sample_fluctuations,
        total_state_expectation_bias_all=total_bias_errors,
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
        raw_l1_norm_total=evolution_data["raw_l1_norm_total"],
        theta=evolution_data["theta"],
        sin_theta=evolution_data["sin_theta"],
        cos_sq=evolution_data["cos_sq"],
        save_trials_list=args.save_trials_list,
    )
    np.savez(out, **output_payload)
    print("saving results to:", out)


if __name__ == "__main__":
    main()
