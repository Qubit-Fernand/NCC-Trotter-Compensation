"""
Action-on-rho prototype for the original channel NCC, keeping only the
leading s=2,3 compensated terms.
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
    commutator,
    pauli_decomposition_stream,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Action-on-rho original channel NCC prototype.")
    parser.add_argument("--N", type=int, default=3, help="number of spins")
    parser.add_argument("--J", type=float, default=1.0, help="interaction strength")
    parser.add_argument("--h", type=float, default=1.0, help="field strength")
    parser.add_argument("--T", type=float, default=1.0, help="total evolution time")
    parser.add_argument("--r", type=int, default=20, help="number of Trotter segments")
    parser.add_argument("--trials", type=int, default=1000, help="Monte Carlo trials")
    parser.add_argument("--seed", type=int, default=7, help="RNG seed")
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
    # Same leading-order channel pairing normalization as in the note:
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
def build_static_data(n: int, j: float, h: float) -> dict:
    """Precompute r-independent operator data for original channel NCC."""
    a_mat, b_mat = build_periodic_ab(n, j, h)
    dim = 2**n
    identity = np.eye(dim, dtype=complex)

    f2 = commutator(b_mat, a_mat)
    f3 = 1j * (2 * commutator(b_mat, commutator(a_mat, b_mat)) + commutator(a_mat, commutator(a_mat, b_mat)))
    f2_coeffs, f2_labels, f2_l1 = pauli_decomposition_stream(f2, antihermitian=True)
    f3_coeffs, f3_labels, f3_l1 = pauli_decomposition_stream(f3, antihermitian=True)

    return {
        "n": n,
        "A_mat": a_mat,
        "B_mat": b_mat,
        "h_total": a_mat + b_mat,
        "identity": identity,
        "dim": dim,
        "F2": f2,
        "F3": f3,
        "f2_coeffs": f2_coeffs,
        "f2_labels": f2_labels,
        "f2_l1": f2_l1,
        "f3_coeffs": f3_coeffs,
        "f3_labels": f3_labels,
        "f3_l1": f3_l1,
    }


def build_tilde_V(static: dict, t_total: float, r: int, validation_tol=1e-10):
    """Build action-on-rho data for one original-NCC step size."""
    a_mat = static["A_mat"]
    b_mat = static["B_mat"]
    h_total = static["h_total"]
    identity = static["identity"]
    dim = static["dim"]

    t = t_total / r
    # Original channel NCC keeps only the leading s=2,3 terms,
    # with the expected BCH/Taylor prefactors 1/2! and 1/3!.
    eta2 = static["f2_l1"] * (t**2) / 2
    eta3 = static["f3_l1"] * (t**3) / 6
    eta_pair_sum = eta2 + eta3
    if eta_pair_sum <= 0:
        raise ValueError("eta_pair_sum must be positive")

    pair_params = paired_channel_parameters(eta_pair_sum)
    pair_scale = pair_params["pair_scale"]

    raw_weights = {
        # Outer order sampling follows eta_s / eta_pair_sum times the common pair_scale.
        2: pair_scale * eta2 / eta_pair_sum,
        3: pair_scale * eta3 / eta_pair_sum,
    }
    raw_l1_norm_total = float(sum(raw_weights.values()))
    p_order = np.array([raw_weights[2] / raw_l1_norm_total, raw_weights[3] / raw_l1_norm_total], dtype=float)

    pair_components = {}
    for order, coeffs, labels, l1_norm in (
        (2, static["f2_coeffs"], static["f2_labels"], static["f2_l1"]),
        (3, static["f3_coeffs"], static["f3_labels"], static["f3_l1"]),
    ):
        probs = np.abs(coeffs) / l1_norm
        Paulis = []
        for prob, coeff, label in zip(probs, coeffs, labels):
            phase = coeff / (1j * abs(coeff))
            w_mat = phase * cached_pauli_matrix_from_label(label)
            hermitian_err = float(np.linalg.norm(w_mat - w_mat.conj().T, ord="fro"))
            if hermitian_err > 1e-10:
                raise ValueError(f"sampled W is not Hermitian (herm_err={hermitian_err:.3e})")
            paired_unitary = pair_params["cos_theta"] * identity + 1j * pair_params["sin_theta"] * w_mat
            Paulis.append(
                {
                    "prob": float(prob),
                    "W_mat": w_mat,
                    "paired_unitary": paired_unitary,
                }
            )
        pair_components[order] = Paulis

    s1 = expm(-1j * b_mat * t) @ expm(-1j * a_mat * t)
    step_exact = expm(-1j * h_total * t)
    u_exact = expm(-1j * h_total * t_total)
    v_exact = step_exact @ expm(1j * a_mat * t) @ expm(1j * b_mat * t)

    def apply_tilde_V_taylor(rho: np.ndarray) -> np.ndarray:
        return rho + (t**2 / 2) * apply_ad_commutator(static["F2"], rho) + (t**3 / 6) * apply_ad_commutator(static["F3"], rho)

    def apply_tilde_V_expectation(rho: np.ndarray) -> np.ndarray:
        out = np.zeros_like(rho)
        for order in (2, 3):
            for Pauli in pair_components[order]:
                # raw_weight_s times the normalized expectation of the paired sampler
                # reconstructs the target I + (t^2/2)F_2 + (t^3/6)F_3 action.
                out += raw_weights[order] * Pauli["prob"] * (rho + eta_pair_sum * apply_ad_commutator(1j * Pauli["W_mat"], rho)) / pair_scale
        return out

    validation_states, _ = computational_basis_density_matrices(static["n"])
    validation_error = max(trace_norm(apply_tilde_V_expectation(rho) - apply_tilde_V_taylor(rho)) for rho in validation_states)
    if validation_error > validation_tol:
        raise ValueError(f"channel tilde_V mismatch between compensation expectation and Taylor action: {validation_error:.3e}")

    return {
        "t": t,
        "S1": s1,
        "step_exact": step_exact,
        "U_exact": u_exact,
        "V_exact": v_exact,
        "eta": {2: eta2, 3: eta3},
        "eta_pair_sum": eta_pair_sum,
        "raw_weights": raw_weights,
        "raw_l1_norm_total": raw_l1_norm_total,
        "p_order": p_order,
        "pair_data": pair_components,
        "pair_orders": (2, 3),
        "validation_error": validation_error,
        "apply_tilde_V_taylor": apply_tilde_V_taylor,
        "apply_tilde_V_expectation": apply_tilde_V_expectation,
        **pair_params,  # contain pair_scale and angles
    }


def sample_channel_then_compensate_descriptor(
    rng: np.random.Generator,
    static: dict,
    evolution_data: dict,
    order: int,
) -> tuple[complex, np.ndarray]:
    """Sample one original compensated remainder channel as a signed unitary descriptor."""
    del static
    Paulis = evolution_data["pair_data"][order]
    probs = np.array([Pauli["prob"] for Pauli in Paulis], dtype=float)
    idx = int(rng.choice(len(Paulis), p=probs))
    Pauli = Paulis[idx]
    if rng.random() < evolution_data["unitary_branch_prob"]:
        return evolution_data["raw_l1_norm_total"], Pauli["paired_unitary"]
    return -evolution_data["raw_l1_norm_total"], Pauli["W_mat"]


def sample_trotter_step_descriptor(
    rng: np.random.Generator,
    static: dict,
    evolution_data: dict,
) -> tuple[complex, np.ndarray]:
    """Sample one full compensated Trotter step as a signed unitary descriptor."""
    order = int(rng.choice((2, 3), p=evolution_data["p_order"]))
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

    static = build_static_data(n=n, j=j, h=h)
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

    print("action-on-rho original channel prototype:", True)
    print("N:", n, "r:", r)
    print("time step:", evolution_data["t"])
    print("eta2, eta3:", evolution_data["eta"][2], evolution_data["eta"][3])
    print("eta_pair_sum:", evolution_data["eta_pair_sum"])
    print("pair theta:", evolution_data["theta"])
    print("pair scale:", evolution_data["pair_scale"])
    print("raw weights:", evolution_data["raw_weights"])
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
    out = data_dir / f"NCC_channel_original_trials{trials}_N{args.N}_r{args.r}.npz"
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
