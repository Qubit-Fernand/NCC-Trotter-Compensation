"""
Dense Liouville-space prototype for the channel version of NCC.

This module mirrors the structure of ``NCC_log.py`` / ``NCC_original.py``, but
works directly with channel superoperators. The implementation follows the PDF
note as closely as possible while staying dense and explicit:

- leading orders (for K=1, namely s=2,3) use the channel-pairing idea from
  Section 3.3;
- higher orders are built in Liouville space from the truncated BCH channel
  expansion, then decomposed in the Pauli basis of the doubled Hilbert space.

Because the channel matrices live in dimension 4^N, this file is intended only
for small-N validation and not for large-system scaling experiments.
"""

import argparse
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.linalg import expm
from tqdm import tqdm

from NCC_log import (
    build_periodic_ab,
    cached_pauli_matrix_from_label,
    pauli_decomposition_stream,
    phi_term,
    tilde_F_term,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Dense channel-space NCC prototype.")
    parser.add_argument("--N", type=int, default=3, help="number of spins")
    parser.add_argument("--J", type=float, default=1.0, help="interaction strength")
    parser.add_argument("--h", type=float, default=1.0, help="field strength")
    parser.add_argument("--T", type=float, default=1.0, help="total evolution time")
    parser.add_argument("--r", type=int, default=20, help="number of Trotter segments")
    parser.add_argument("--trials", type=int, default=1000, help="Monte Carlo trials")
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
        help="safety guard: dense channel construction is only practical for very small N",
    )
    parser.add_argument(
        "--save_trials_list",
        action="store_true",
        help="store per-trial sampled channel matrices in the output npz (small N only)",
    )
    return parser.parse_args()


def channel_from_unitary(unitary: np.ndarray) -> np.ndarray:
    """Return the Liouville superoperator of rho -> U rho U^\dagger."""
    return np.kron(unitary.conj(), unitary)


def ad_commutator_from_operator(operator: np.ndarray) -> np.ndarray:
    """Return the superoperator ad_operator: rho -> operator rho - rho operator."""
    dim = operator.shape[0]
    identity = np.eye(dim, dtype=complex)
    # act on vectorized density matrix as (I \otimes operator - operator^T \otimes I) vec(rho).
    return np.kron(identity, operator) - np.kron(operator.T, identity)


def paired_channel_parameters(eta_pair_sum: float) -> dict:
    """Solve sin(theta) / cos(theta)^2 = eta_pair_sum and return the pairing data."""
    if eta_pair_sum <= 0:
        return {
            "theta": 0.0,
            "sin_theta": 0.0,
            "cos_sq": 1.0,
            "pair_scale": 1.0,
        }

    # Stable form of (sqrt(1 + 4 eta^2) - 1) / (2 eta).
    sin_theta = (2.0 * eta_pair_sum) / (math.sqrt(1.0 + 4.0 * eta_pair_sum**2) + 1.0)
    cos_sq = sin_theta / eta_pair_sum
    theta = math.asin(min(1.0, max(-1.0, sin_theta)))
    pair_scale = (1.0 + sin_theta**2) / cos_sq
    return {
        "theta": theta,
        "sin_theta": sin_theta,
        "cos_sq": cos_sq,
        "pair_scale": pair_scale,
    }


@lru_cache(maxsize=None)
def build_static_data(n, q0, s0, epsilon=0.01, j=1.0, h=1.0, K=1, max_dense_qubits=3):
    """Precompute r-independent operator and channel data for dense channel NCC."""
    if n > max_dense_qubits:
        raise ValueError(f"dense channel prototype only supports N <= {max_dense_qubits}; " f"received N={n}. For larger N, a locality-based sampler is needed instead.")

    A_mat, B_mat = build_periodic_ab(n, j, h)
    dim = 2**n
    identity = np.eye(dim, dtype=complex)
    identity_super = np.eye(dim * dim, dtype=complex)

    # [ad_A, ad_B] = ad_[A,B], phi can be built in this way.
    Phi_terms = phi_term(A_mat, B_mat, q0)
    tilde_F_operator_terms = tilde_F_term(Phi_terms, K, q0, s0)

    Phi_channel_terms = {order: ad_commutator_from_operator(Phi_terms[order]) for order in range(K + 1, q0 + 1)}
    tilde_F_channel_terms = tilde_F_term(Phi_channel_terms, K, q0, s0)

    F_terms = {}
    pairable_orders = []
    tail_orders = []
    pair_validation_errors = {}

    for order in range(K + 1, s0 + 1):
        if order <= 2 * K + 1:
            operator_term = tilde_F_operator_terms[order]
            antiherm_err = float(np.linalg.norm(operator_term + operator_term.conj().T, ord="fro"))
            if antiherm_err > 1e-8:
                raise ValueError(f"leading order s={order} is not anti-Hermitian enough for channel pairing " f"(antiherm_err={antiherm_err:.3e})")

            coeffs, labels, l1_norm = pauli_decomposition_stream(
                operator_term,
                antihermitian=True,
            )
            pair_validation = float(
                np.linalg.norm(
                    tilde_F_channel_terms[order] - ad_commutator_from_operator(operator_term),
                    2,
                )
            )
            pair_validation_errors[order] = pair_validation
            pairable_orders.append(order)
            F_terms[order] = {
                "kind": "pair",
                "coeffs": coeffs,
                "labels": labels,
                "l1_norm": l1_norm,
                "antiherm_err": antiherm_err,
            }
        else:
            coeffs, labels, l1_norm = pauli_decomposition_stream(tilde_F_channel_terms[order])
            tail_orders.append(order)
            F_terms[order] = {
                "kind": "tail",
                "coeffs": coeffs,
                "labels": labels,
                "l1_norm": l1_norm,
            }

    return {
        "A_mat": A_mat,
        "B_mat": B_mat,
        "h_total": A_mat + B_mat,
        "identity": identity,
        "identity_super": identity_super,
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


def build_tilde_V(static, t_total, r, validation_tol=1e-8):
    """Build the dense single-step compensation superoperator and bias diagnostics."""
    A_mat = static["A_mat"]
    B_mat = static["B_mat"]
    h_total = static["h_total"]
    identity_super = static["identity_super"]
    F_terms = static["F_terms"]
    s_orders = static["s_orders"]
    pairable_orders = static["pairable_orders"]
    tail_orders = static["tail_orders"]
    tilde_F_channel_terms = static["tilde_F_channel_terms"]

    t = t_total / r
    s1 = expm(-1j * B_mat * t) @ expm(-1j * A_mat * t)
    s1_channel = channel_from_unitary(s1)
    step_exact = expm(-1j * h_total * t)
    step_exact_channel = channel_from_unitary(step_exact)
    U_exact = expm(-1j * h_total * t_total)
    U_exact_channel = channel_from_unitary(U_exact)
    V_exact = step_exact @ expm(1j * A_mat * t) @ expm(1j * B_mat * t)
    V_exact_channel = channel_from_unitary(V_exact)

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

    tilde_V = np.zeros_like(identity_super)
    for order in s_orders:
        data = F_terms[order]
        weight = raw_weights[order]
        probs = np.abs(data["coeffs"]) / data["l1_norm"]
        for prob, coeff, label in zip(probs, data["coeffs"], data["labels"]):
            if data["kind"] == "pair":
                phase = coeff / (1j * abs(coeff))
                W_mat = phase * cached_pauli_matrix_from_label(label)
                ad_iW = ad_commutator_from_operator(1j * W_mat)
                tilde_V += weight * prob * ((identity_super + eta_pair_sum * ad_iW) / pair_scale)
            else:
                tilde_V += weight * prob * ((coeff / abs(coeff)) * cached_pauli_matrix_from_label(label))

    tilde_V_taylor = identity_super.copy()
    for order in s_orders:
        tilde_V_taylor += tilde_F_channel_terms[order] * (t**order)

    validation_error = float(np.linalg.norm(tilde_V - tilde_V_taylor, 2))
    if validation_error > validation_tol:
        raise ValueError(f"channel tilde_V mismatch between compensation sum and Taylor sum: {validation_error:.3e}")

    deterministic = np.linalg.matrix_power(tilde_V @ s1_channel, r)
    deterministic_bias = float(np.linalg.norm(deterministic - U_exact_channel, 2))

    return {
        "t": t,
        "s1": s1,
        "s1_channel": s1_channel,
        "step_exact_channel": step_exact_channel,
        "U_exact_channel": U_exact_channel,
        "V_exact_channel": V_exact_channel,
        "tilde_V": tilde_V,
        "deterministic": deterministic,
        "deterministic_bias": deterministic_bias,
        "validation_error": validation_error,
        "eta": eta,
        "eta_pair_sum": eta_pair_sum,
        "pair_scale": pair_scale,
        "raw_weights": raw_weights,
        "raw_total": raw_total,
        **pair_data,
    }


def sample_channel_then_compensate(
    rng: np.random.Generator,
    static: dict,
    evolution_data: dict,
    order: int,
    atol: float = 1e-10,
):
    """Sample one dense compensated channel component."""
    data = static["F_terms"][order]
    probs = np.abs(data["coeffs"]) / data["l1_norm"]
    idx = int(rng.choice(len(data["labels"]), p=probs))
    coeff = data["coeffs"][idx]

    if data["kind"] == "pair":
        phase = coeff / (1j * abs(coeff))
        W_mat = phase * cached_pauli_matrix_from_label(data["labels"][idx])
        hermitian_err = np.linalg.norm(W_mat - W_mat.conj().T, ord="fro")
        if hermitian_err > atol:
            raise ValueError(f"sampled W is not Hermitian (herm_err={hermitian_err:.3e})")
        ad_iW = ad_commutator_from_operator(1j * W_mat)
        return evolution_data["raw_total"] * ((static["identity_super"] + evolution_data["eta_pair_sum"] * ad_iW) / evolution_data["pair_scale"])

    basis_super = cached_pauli_matrix_from_label(data["labels"][idx])
    return evolution_data["raw_total"] * ((coeff / abs(coeff)) * basis_super)


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

    print("dense channel prototype:", True)
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

    rng = np.random.default_rng(seed=7)
    p_order = np.array(
        [evolution_data["raw_weights"][order] / evolution_data["raw_total"] for order in static["s_orders"]],
        dtype=float,
    )

    def single_step_channel_sampling(num_trials):
        V_list = [] if args.save_trials_list else None
        average = np.zeros_like(static["identity_super"])
        for _ in tqdm(range(num_trials), desc="single-step channel trials"):
            order = int(rng.choice(static["s_orders"], p=p_order))
            sample = sample_channel_then_compensate(rng, static, evolution_data, order)
            average += sample
            if V_list is not None:
                V_list.append(sample)
        average /= num_trials
        return V_list, average

    single_step_error_before = np.linalg.norm(
        evolution_data["s1_channel"] - evolution_data["step_exact_channel"],
        2,
    )
    print("single-step channel error before:", single_step_error_before)

    V_list, V_avg = single_step_channel_sampling(trials)
    single_step_fluctuation = np.linalg.norm(V_avg - evolution_data["tilde_V"], 2)
    single_step_sample_error = np.linalg.norm(V_avg - evolution_data["V_exact_channel"], 2)
    single_step_expectation_bias = np.linalg.norm(evolution_data["tilde_V"] - evolution_data["V_exact_channel"], 2)

    print("single-step channel sample error after compensation:", single_step_sample_error)
    print("single-step channel sample fluctuation:", single_step_fluctuation)
    print("single-step channel expectation bias:", single_step_expectation_bias)

    def multi_step_channel_sampling(num_trials):
        U_total_list = [] if args.save_trials_list else None
        average = np.zeros_like(static["identity_super"])
        for _ in tqdm(range(num_trials), desc="multi-step channel trials"):
            evo = static["identity_super"].copy()
            for _ in range(r):
                order = int(rng.choice(static["s_orders"], p=p_order))
                evo = sample_channel_then_compensate(rng, static, evolution_data, order) @ evolution_data["s1_channel"] @ evo
            average += evo
            if U_total_list is not None:
                U_total_list.append(evo)
        average /= num_trials
        return U_total_list, average

    total_error_before = np.linalg.norm(
        np.linalg.matrix_power(evolution_data["s1_channel"], r) - evolution_data["U_exact_channel"],
        2,
    )
    print("total channel error before:", total_error_before)

    U_total_list, evo_avg = multi_step_channel_sampling(trials)
    total_sample_fluctuation = np.linalg.norm(evo_avg - evolution_data["deterministic"], 2)
    total_sample_error = np.linalg.norm(evo_avg - evolution_data["U_exact_channel"], 2)
    total_expectation_bias = evolution_data["deterministic_bias"]

    print("multi-step channel sample error after compensation:", total_sample_error)
    print("multi-step channel sample fluctuation:", total_sample_fluctuation)
    print("multi-step channel expectation bias:", total_expectation_bias)

    data_dir = Path("data/no_search")
    data_dir.mkdir(parents=True, exist_ok=True)
    out = data_dir / f"results_channel_trials{trials}_q{q0}_s{s0}.npz"

    output_payload = dict(
        S=evolution_data["s1_channel"],
        V_tilde=evolution_data["tilde_V"],
        V_exact=evolution_data["V_exact_channel"],
        V_average=V_avg,
        U_total_average=evo_avg,
        single_step_error_before=single_step_error_before,
        single_step_sample_error=single_step_sample_error,
        single_step_fluctuation=single_step_fluctuation,
        single_step_expectation_bias=single_step_expectation_bias,
        total_error_before=total_error_before,
        total_sample_error=total_sample_error,
        total_sample_fluctuation=total_sample_fluctuation,
        total_expectation_bias=total_expectation_bias,
        N=n,
        J=j,
        h=h,
        T=t_total,
        r=r,
        trials=trials,
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
        output_payload["V_list"] = np.stack(V_list)
        output_payload["U_total_list"] = np.stack(U_total_list)

    np.savez(out, **output_payload)
    print("saving results to:", out)


if __name__ == "__main__":
    main()
