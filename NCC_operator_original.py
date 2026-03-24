"""
We first compute the commutator and then calculate its Pauli 1-norm.
We use the commutator results' Pauli expansion to pair into exp, unlike the layer-wise pair in pseudocode in prx paper.
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.linalg import expm
from tqdm import tqdm

from Pauli_Hamiltonian_BCH import (
    build_periodic_ab,
    cached_pauli_matrix_from_label,
    commutator,
    pauli_decomposition_stream,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=6, help="number of spins")
    parser.add_argument("--J", type=float, default=1.0, help="interaction strength")
    parser.add_argument("--h", type=float, default=1.0, help="transverse field strength")
    parser.add_argument("--T", type=float, default=1.0, help="evolution time")
    parser.add_argument("--r", type=int, default=20, help="Trotter steps")
    parser.add_argument("--trials", type=int, default=1000, help="NCC trials")
    parser.add_argument(
        "--save_trials_list",
        action="store_true",
        help="store per-trial sampled matrices V_list and U_total_list in memory and output npz",
    )
    return parser.parse_args()


def build_static_data(n: int, j: float, h: float) -> dict:
    A_mat, B_mat = build_periodic_ab(n, j, h)
    c1 = commutator(B_mat, A_mat)
    c2 = 1j * (2 * commutator(B_mat, commutator(A_mat, B_mat)) + commutator(A_mat, commutator(A_mat, B_mat)))
    c1_coeffs, c1_labels, c1_l1 = pauli_decomposition_stream(c1, antihermitian=True)
    c2_coeffs, c2_labels, c2_l1 = pauli_decomposition_stream(c2, antihermitian=True)
    dim = 2**n
    return {
        "n": n,
        "A_mat": A_mat,
        "B_mat": B_mat,
        "C1": c1,
        "C2": c2,
        "c1_coeffs": c1_coeffs,
        "c1_labels": c1_labels,
        "c1_l1": c1_l1,
        "c2_coeffs": c2_coeffs,
        "c2_labels": c2_labels,
        "c2_l1": c2_l1,
        "identity": np.eye(dim, dtype=complex),
        "h_total": A_mat + B_mat,
    }


def build_tilde_V(static, t_total: float, r: int, validation_tol=1e-10):
    """Build tilde_V and deterministic bias for the original-NCC step."""
    t = t_total / r
    eta2 = static["c1_l1"] * (t**2) / 2
    eta3 = static["c2_l1"] * (t**3) / 6
    eta_sum = eta2 + eta3
    if eta_sum <= 0:
        raise ValueError("eta_sum must be positive")
    p_s = np.array([eta2 / eta_sum, eta3 / eta_sum], dtype=float)
    S1 = expm(-1j * static["B_mat"] * t) @ expm(-1j * static["A_mat"] * t)
    U_exact = expm(-1j * static["h_total"] * t_total)

    tilde_V = np.zeros_like(static["identity"])
    # actual coeffs and l1_norm need to multiply the factor x^s/s!
    for weight, coeffs, labels, l1_norm in (
        (p_s[0], static["c1_coeffs"], static["c1_labels"], static["c1_l1"]),
        (p_s[1], static["c2_coeffs"], static["c2_labels"], static["c2_l1"]),
    ):
        # actually prob = coeffs * x^s/s! / (l1_norm * x^s/s!)
        probs = np.abs(coeffs) / l1_norm
        for prob, coeff, label in zip(probs, coeffs, labels):
            sign = coeff / (1j * abs(coeff))
            W_mat = sign * cached_pauli_matrix_from_label(label)
            tilde_V += weight * prob * (static["identity"] + 1j * eta_sum * W_mat)

    # Taylor form for original NCC: I + C1 t^2 / 2 + C2 t^3 / 6.
    tilde_V_taylor = static["identity"] + static["C1"] * (t**2) / 2 + static["C2"] * (t**3) / 6
    validation_error = float(np.linalg.norm(tilde_V - tilde_V_taylor, 2))
    if validation_error > validation_tol:
        raise ValueError(f"tilde_V mismatch between compensation sum and Taylor sum: {validation_error:.3e}")

    deterministic = np.linalg.matrix_power(tilde_V @ S1, r)
    deterministic_bias = float(np.linalg.norm(deterministic - U_exact, 2))

    return {
        "t": t,
        "p_s": p_s,
        "eta_sum": eta_sum,
        "eta2": eta2,
        "eta3": eta3,
        "S1": S1,
        "U_exact": U_exact,
        "tilde_V": tilde_V,
        # "tilde_V_taylor": tilde_V_taylor,
        "deterministic": deterministic,
        "deterministic_bias": deterministic_bias,
        "validation_error": validation_error,
    }


def sample_Pauli_then_compensate_exp(
    rng: np.random.Generator,
    static: dict,
    evolution_data: dict,
    order: int | None = None,
    atol: float = 1e-10,
) -> np.ndarray:
    """Sample one full compensation component for original-NCC."""
    if order is None:
        order = int(rng.choice([2, 3], p=evolution_data["p_s"]))
    if order == 2:
        coeffs = static["c1_coeffs"]
        labels = static["c1_labels"]
        l1_norm = static["c1_l1"]
    elif order == 3:
        coeffs = static["c2_coeffs"]
        labels = static["c2_labels"]
        l1_norm = static["c2_l1"]
    else:
        raise ValueError(f"unsupported order s={order}")
    probs = np.abs(coeffs) / l1_norm
    # Sample one Pauli from the order-s commutator expansion.
    idx = int(rng.choice(len(labels), p=probs))
    coeff = coeffs[idx]
    # With this probability, sample I + eta_sum * (\pm i) * P.
    sign = coeff / (1j * abs(coeff))
    W_mat = sign * cached_pauli_matrix_from_label(labels[idx])
    # For anti-Hermitian F_2 and F_3, the coefficients are purely imaginary,
    # so the sampled W should be Hermitian.
    hermitian_err = np.linalg.norm(W_mat - W_mat.conj().T, ord="fro")
    if hermitian_err > atol:
        raise ValueError(f"sampled W is not Hermitian (herm_err={hermitian_err:.3e})")
    # This is the pre-pairing form I + i eta_sum W, i.e. the full compensated
    # component already including the original-mode normalization convention.
    return static["identity"] + 1j * evolution_data["eta_sum"] * W_mat


def main():
    args = parse_args()

    # Parameters
    N = args.N
    J = args.J
    h = args.h
    T = args.T
    r = args.r

    g = 2 * (J + h)  # extensive parameter
    t = T / r  # step size
    K = 1  # order of product formula, fixed to 1 for original NCC

    print("time step:", t)

    static = build_static_data(N, J, h)
    evolution_data = build_tilde_V(static, T, r)
    A = static["A_mat"]
    B = static["B_mat"]

    # note the order of exp(A) and exp(B)
    S = evolution_data["S1"]
    V_exact = expm(-1j * static["h_total"] * t) @ expm(1j * A * t) @ expm(1j * B * t)
    tilde_V = evolution_data["tilde_V"]
    S_r = np.linalg.matrix_power(S, r)
    evolution_exact = evolution_data["U_exact"]
    identity = static["identity"]

    print("C1 l1:", static["c1_l1"])
    print("C2 l1:", static["c2_l1"])
    print("C1/C2 Pauli terms:", len(static["c1_labels"]), len(static["c2_labels"]))
    print("eta2, eta3:", evolution_data["eta2"], evolution_data["eta3"])
    print("p_s:", evolution_data["p_s"])
    print("tilde_V compensation-vs-Taylor check:", evolution_data["validation_error"])

    # K = 1, sample from s = 2, 3
    def NCC_sampling(trials):
        rng = np.random.default_rng(seed=7)

        # NCC sampling
        s_list = [2, 3]

        V_list = [] if args.save_trials_list else None
        V_average = np.zeros_like(identity)
        for _ in tqdm(range(trials), desc="single step trials"):
            s = int(rng.choice(s_list, p=evolution_data["p_s"]))
            sample = sample_Pauli_then_compensate_exp(rng, static, evolution_data, order=s)
            V_average += sample
            if V_list is not None:
                V_list.append(sample)

        V_average /= trials

        return V_list, V_average

    single_step_error_before = np.linalg.norm(S - expm(-1j * static["h_total"] * t), 2)
    print("single Trotter step error before compensation:\n", single_step_error_before)

    V_list, V_average = NCC_sampling(trials=args.trials)
    single_step_error_after = np.linalg.norm(V_average - V_exact, 2)
    single_step_fluctuation = np.linalg.norm(V_average - tilde_V, 2)
    single_step_bias = np.linalg.norm(tilde_V - V_exact, 2)

    print("single step error after compensation:\n", single_step_error_after)
    print("single step fluctuation:\n", single_step_fluctuation)
    print("single step expectation bias:\n", single_step_bias)

    def multi_step_NCC_sampling(trials):
        rng = np.random.default_rng(seed=7)

        # NCC sampling
        s_list = [2, 3]

        U_total_list = [] if args.save_trials_list else None
        U_total_average = np.zeros_like(identity)
        for _ in tqdm(range(trials), desc="multi step trials"):
            evolution = identity.copy()
            for _ in range(r):
                s = int(rng.choice(s_list, p=evolution_data["p_s"]))
                evolution = sample_Pauli_then_compensate_exp(rng, static, evolution_data, order=s) @ S @ evolution

            U_total_average += evolution
            if U_total_list is not None:
                U_total_list.append(evolution)

        U_total_average /= trials

        return U_total_list, U_total_average

    total_error_before = np.linalg.norm(S_r - evolution_exact, 2)
    print("total evolution error before compensation:\n", total_error_before)

    U_total_list, U_total_average = multi_step_NCC_sampling(trials=args.trials)
    total_error_after = np.linalg.norm(U_total_average - evolution_exact, 2)
    total_fluctuation = np.linalg.norm(U_total_average - evolution_data["deterministic"], 2)
    total_bias = evolution_data["deterministic_bias"]

    print("total evolution error after compensation:\n", total_error_after)
    print("total evolution fluctuation:\n", total_fluctuation)
    print("total evolution expectation bias:\n", total_bias)

    data_dir = Path("data/no_search")
    data_dir.mkdir(parents=True, exist_ok=True)
    output_path = data_dir / f"NCC_original_trials{args.trials}_N{args.N}_r{args.r}.npz"
    print("saving results to:", output_path)
    output_payload = dict(
        A=A,
        B=B,
        S=S,
        V_tilde=tilde_V,
        V_exact=V_exact,
        V_average=V_average,
        U_total_average=U_total_average,
        single_step_error_before=single_step_error_before,
        single_step_error_after=single_step_error_after,
        single_step_fluctuation=single_step_fluctuation,
        single_step_bias=single_step_bias,
        total_error_before=total_error_before,
        total_error_after=total_error_after,
        total_fluctuation=total_fluctuation,
        total_bias=total_bias,
        N=N,
        J=J,
        h=h,
        T=T,
        r=r,
        trials=args.trials,
        save_trials_list=args.save_trials_list,
    )
    if args.save_trials_list:
        output_payload["V_list"] = np.stack(V_list)
        output_payload["U_total_list"] = np.stack(U_total_list)
    np.savez(output_path, **output_payload)


if __name__ == "__main__":
    main()
