"""Phi_q and each tilde_F term are separated from x^s as fixed matrics."""

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
    commutator,
    phi_term,
    pauli_decomposition_stream,
    tilde_F_term,
)


def parse_args():
    parser = argparse.ArgumentParser(description="NCC log-precision prototype with periodic-boundary simplification.")
    parser.add_argument("--N", type=int, default=6, help="number of spins")
    parser.add_argument("--J", type=float, default=1.0, help="interaction strength")
    parser.add_argument("--h", type=float, default=1.0, help="field strength")
    parser.add_argument("--T", type=float, default=1.0, help="total evolution time")
    parser.add_argument("--r", type=int, default=20, help="number of Trotter segments")
    parser.add_argument("--trials", type=int, default=1000, help="Monte Carlo trials")
    parser.add_argument(
        "--save_trials_list",
        action="store_true",
        help="store per-trial sampled matrices V_list and U_total_list in memory and output npz",
    )
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
    return parser.parse_args()


@lru_cache(maxsize=None)
def build_static_data(n, q0, s0, j=1.0, h=1.0, K=1):
    """Precompute r-independent data for log-NCC evaluation."""

    A_mat, B_mat = build_periodic_ab(n, j, h)
    Phi_terms = phi_term(A_mat, B_mat, q0)
    tilde_F_terms = tilde_F_term(Phi_terms, K, q0, s0)
    identity = np.eye(2**n, dtype=complex)

    # F_terms is the Pauli decomposition of the tilde_F_terms, tagged pairable or not.
    F_terms = {}
    tilde_F_l1 = {}
    pairable_orders = []
    non_pairable_orders = []
    leading_orders = list(range(K + 1, min(s0, 2 * K + 1) + 1))
    for order in range(K + 1, s0 + 1):
        coeffs, labels, l1_norm = pauli_decomposition_stream(tilde_F_terms[order])
        F_terms[order] = {"kind": "tail", "coeffs": coeffs, "labels": labels, "l1_norm": l1_norm}
        tilde_F_l1[order] = l1_norm

        if order > 2 * K + 1:
            continue

        antiherm_err = np.linalg.norm(
            tilde_F_terms[order] + tilde_F_terms[order].conj().T,
            ord="fro",
        )
        if antiherm_err <= 1e-8:
            pair_coeffs, pair_labels, _ = pauli_decomposition_stream(
                tilde_F_terms[order],
                antihermitian=True,
            )
            pairable_orders.append(order)
            F_terms[order] = {
                "kind": "pair",
                "coeffs": pair_coeffs,
                "labels": pair_labels,
                "l1_norm": l1_norm,
            }
        else:
            non_pairable_orders.append((order, float(antiherm_err)))

    return {
        "A_mat": A_mat,
        "B_mat": B_mat,
        "identity": identity,
        "Phi_terms": Phi_terms,
        "tilde_F_terms": tilde_F_terms,
        "tilde_F_l1": tilde_F_l1,
        "F_terms": F_terms,
        "s_orders": list(range(K + 1, s0 + 1)),
        "leading_orders": leading_orders,
        "pairable_orders": pairable_orders,
        "non_pairable_orders": non_pairable_orders,
        "K": K,
        "q0": q0,
        "s0": s0,
    }


def build_tilde_V(
    static, t_total, r, validation_tol=7 * 1e-9
):  # 1e-10 is too tight for some cases, loosen to 7e-9 to avoid false alarm in log-NCC validation. The main point is to check the consistency between the compensation sum and the Taylor expansion, not to get a very tight match between them, since they are derived from different formulas and numerical errors can be amplified in different ways.
    """Build tilde_V and the associated deterministic bias data for one step size."""
    A_mat = static["A_mat"]
    B_mat = static["B_mat"]
    identity = static["identity"]
    s_orders = static["s_orders"]
    F_terms = static["F_terms"]
    tilde_F_terms = static["tilde_F_terms"]
    K = static["K"]

    t = t_total / r
    S1 = expm(-1j * B_mat * t) @ expm(-1j * A_mat * t)
    U_exact = expm(-1j * (A_mat + B_mat) * t_total)

    eta = {order: F_terms[order]["l1_norm"] * (t**order) for order in s_orders}
    leading_orders = [s for s in s_orders if s <= 2 * K + 1]
    tail_orders = [s for s in s_orders if s > 2 * K + 1]
    # eta_sum = eta_2 + eta_3
    eta_pair_sum = sum(eta[s] for s in leading_orders)
    # sqrt(1 + eta_sum^2) is the normalization factor for the pair compensation term, which is the dominant contribution to tilde_V when eta_sum is large.
    pair_scale = math.sqrt(1.0 + eta_pair_sum**2)
    raw_weights = {}
    if eta_pair_sum > 0:
        for s in leading_orders:
            raw_weights[s] = pair_scale * eta[s] / eta_pair_sum
    for s in tail_orders:
        raw_weights[s] = eta[s]

    tilde_V = np.zeros_like(identity)
    for order in s_orders:
        F_term = F_terms[order]
        weight = raw_weights[order]
        probs = np.abs(F_term["coeffs"]) / F_term["l1_norm"]
        for prob, coeff, label in zip(probs, F_term["coeffs"], F_term["labels"]):
            pauli = cached_pauli_matrix_from_label(label)
            if F_term["kind"] == "pair":
                phase = coeff / (1j * abs(coeff))
                W_mat = phase * pauli
                tilde_V += weight * prob * ((identity + 1j * eta_pair_sum * W_mat) / pair_scale)
            else:
                tilde_V += weight * prob * (coeff / abs(coeff) * pauli)

    # another way to compute tilde_V to check consistency
    tilde_V_taylor = identity.copy()
    for order in s_orders:
        tilde_V_taylor += tilde_F_terms[order] * (t**order)
    validation_error = float(np.linalg.norm(tilde_V - tilde_V_taylor, 2))
    if validation_error > validation_tol:
        raise ValueError(f"tilde_V mismatch between compensation sum and Taylor sum: {validation_error:.3e}")

    # This is the total deterministic bias for r repeated compensated steps,
    # not the single-step tilde_V object built above.
    deterministic = np.linalg.matrix_power(tilde_V @ S1, r)
    deterministic_bias = float(np.linalg.norm(deterministic - U_exact, 2))

    return {
        "tilde_V": tilde_V,
        "S1": S1,
        "U_exact": U_exact,
        "deterministic": deterministic,
        "deterministic_bias": deterministic_bias,
        "validation_error": validation_error,
        "eta": eta,
        "eta_pair_sum": eta_pair_sum,
        "pair_scale": pair_scale,
        "raw_weights": raw_weights,
    }


def main():
    args = parse_args()
    n = args.N
    j = args.J
    h = args.h
    t_total = args.T
    r = args.r
    trials = args.trials
    epsilon = args.epsilon
    t = t_total / r

    print("time step:", t)

    k_local = 2
    g = 2 * (j + h)
    a_max = 1.0
    K = 1
    if K == 1:
        kappa = 1
    else:
        kappa = 2 * (5 ** (np.ceil(K / 2) - 1))

    # q0/s0 and convergence checks from NCC_with_log_precision.pdf
    coeff = a_max * kappa + 1
    s0 = int(np.ceil(np.log(4 / epsilon))) if args.s0 <= 0 else args.s0
    s0 = max(3, s0)
    q0 = int(np.ceil(np.log(2 * n / epsilon))) if args.q0 <= 0 else args.q0
    q0 = max(3, q0)
    lambda_comm = 4 * coeff * q0 * k_local * g * (2 * n) ** (1 / 2)

    cond_bch_truncation = 8 * math.e * k_local * q0 * coeff * g * t
    cond_finite_s_truncation = math.e * lambda_comm * t
    print("q0:", q0, "s0:", s0, "lambda_comm:", lambda_comm)
    print("Lemma 3 condition 8e(a_max*kappa+1)q0kg t <= 1:", cond_bch_truncation)
    print("Lemma 5 condition e*lambda_comm*t <= 1:", cond_finite_s_truncation)

    static = build_static_data(n=n, q0=q0, s0=s0, j=j, h=h, K=K)
    evolution_data = build_tilde_V(static, t_total, r)
    A_mat = static["A_mat"]
    B_mat = static["B_mat"]
    identity = static["identity"]
    Phi_terms = static["Phi_terms"]
    tilde_F_terms = static["tilde_F_terms"]
    tilde_F_l1 = static["tilde_F_l1"]
    F_terms = static["F_terms"]
    s_orders = static["s_orders"]
    pairable_orders = static["pairable_orders"]
    non_pairable_orders = static["non_pairable_orders"]
    S1 = evolution_data["S1"]
    tilde_V = evolution_data["tilde_V"]
    U_exact = evolution_data["U_exact"]
    eta = evolution_data["eta"]
    eta_pair_sum = evolution_data["eta_pair_sum"]
    raw_weights = evolution_data["raw_weights"]
    V_exact = expm(-1j * (A_mat + B_mat) * t) @ expm(1j * A_mat * t) @ expm(1j * B_mat * t)

    print("Phi extraction method:", "direct BCH commutator formula")
    raw_total = float(sum(raw_weights.values()))
    p_order = np.array([raw_weights[s] / raw_total for s in s_orders], dtype=float)

    print("eta_pair_sum:", eta_pair_sum)
    print("eta by order:", {s: eta[s] for s in s_orders})
    print("tilde_F Pauli-l1:", {s: tilde_F_l1[s] for s in s_orders})
    print("mixed raw weights:", {s: raw_weights[s] for s in s_orders})
    print("Euler-pairable orders:", pairable_orders)
    if non_pairable_orders:
        print("non-pairable orders (anti-Hermitian defect):", non_pairable_orders)
    print("tilde_V compensation-vs-Taylor check:", evolution_data["validation_error"])

    rng = np.random.default_rng(seed=7)

    def sample_Pauli_then_compensate_exp(order, atol=1e-10):
        """Sample one full compensation component, including the raw-weight sum."""
        F_term = F_terms[order]
        labels = F_term["labels"]
        probs = np.abs(F_term["coeffs"]) / F_term["l1_norm"]
        # sample Pauli from the expansion of given F_term
        idx = int(rng.choice(len(labels), p=probs))
        coeff = F_term["coeffs"][idx]
        pauli = cached_pauli_matrix_from_label(labels[idx])
        if F_term["kind"] == "pair":
            phase = coeff / (1j * abs(coeff))
            W_mat = phase * pauli
            hermitian_err = np.linalg.norm(W_mat - W_mat.conj().T, ord="fro")
            if hermitian_err > atol:
                raise ValueError(f"sampled W is not Hermitian (herm_err={hermitian_err:.3e})")

            # apply exp(i theta W), you must multiply total 1-norm raw_total.
            return raw_total * ((identity + 1j * eta_pair_sum * W_mat) / evolution_data["pair_scale"])
        # apply W like (1+i)X, you must multiply total 1-norm raw_total.
        return raw_total * (coeff / abs(coeff) * pauli)

    # Sampling start here
    def NCC_sampling(num_trials):
        V_list = [] if args.save_trials_list else None
        v_average = np.zeros_like(identity)
        for _ in tqdm(range(num_trials), desc="single step trials"):
            order = int(rng.choice(s_orders, p=p_order))
            sample = sample_Pauli_then_compensate_exp(order)
            v_average += sample
            if V_list is not None:
                V_list.append(sample)
        v_average /= num_trials
        return V_list, v_average

    single_step_error_before = np.linalg.norm(S1 - expm(-1j * (A_mat + B_mat) * t), 2)
    print("single-step error before:", single_step_error_before)

    V_list, V_avg = NCC_sampling(trials)
    single_step_fluctuation = np.linalg.norm(V_avg - tilde_V, 2)
    single_step_sample_error = np.linalg.norm(V_avg - V_exact, 2)
    single_step_expectation_bias = np.linalg.norm(tilde_V - V_exact, 2)

    print("single-step sample error after compensation:", single_step_sample_error)
    print("single-step sample fluctuation:", single_step_fluctuation)
    print("single-step expectation bias:", single_step_expectation_bias)

    def multi_step_NCC_sampling(num_trials):
        U_total_list = [] if args.save_trials_list else None
        U_total_average = np.zeros_like(identity)
        for _ in tqdm(range(num_trials), desc="multi step trials"):
            evo = np.eye(2**n, dtype=complex)
            for _ in range(r):
                order = int(rng.choice(s_orders, p=p_order))
                evo = sample_Pauli_then_compensate_exp(order) @ S1 @ evo
            U_total_average += evo
            if U_total_list is not None:
                U_total_list.append(evo)
        U_total_average /= num_trials
        return U_total_list, U_total_average

    total_error_before = np.linalg.norm(np.linalg.matrix_power(S1, r) - U_exact, 2)
    print("total error before:", total_error_before)

    U_total_list, evo_avg = multi_step_NCC_sampling(trials)

    total_sample_fluctuation = np.linalg.norm(evo_avg - evolution_data["deterministic"], 2)
    total_sample_error = np.linalg.norm(evo_avg - U_exact, 2)
    total_expectation_bias = evolution_data["deterministic_bias"]

    print("multi-step sample error after compensation:", total_sample_error)
    print("multi-step sample fluctuation:", total_sample_fluctuation)
    print("multi-step expectation bias:", total_expectation_bias)

    data_dir = Path("data/no_search")
    data_dir.mkdir(parents=True, exist_ok=True)
    out = data_dir / f"NCC_log_trials{trials}_N{args.N}_r{args.r}_q{q0}_s{s0}.npz"
    output_payload = dict(
        A=A_mat,
        B=B_mat,
        S=S1,
        V_tilde=tilde_V,
        V_exact=V_exact,
        V_average=V_avg,
        U_total_average=evo_avg,
        orders=np.array(s_orders, dtype=int),
        phi_matrices=np.stack([Phi_terms[s] for s in s_orders]),
        tilde_F_matrices=np.stack([tilde_F_terms[s] for s in s_orders]),
        single_step_error_before=single_step_error_before,
        single_step_sample_error=single_step_sample_error,
        single_step_fluctuation=single_step_fluctuation,
        single_step_expectation_bias=single_step_expectation_bias,
        total_error_before=total_error_before,
        total_sample_fluctuation=total_sample_fluctuation,
        total_sample_error=total_sample_error,
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
        eta_sum=eta_pair_sum,
        raw_total=raw_total,
        cond_bch_truncation=cond_bch_truncation,
        cond_finite_s_truncation=cond_finite_s_truncation,
        save_trials_list=args.save_trials_list,
    )
    if args.save_trials_list:
        output_payload["V_list"] = np.stack(V_list)
        output_payload["U_total_list"] = np.stack(U_total_list)
    np.savez(out, **output_payload)
    print("saving results to:", out)


if __name__ == "__main__":
    main()
