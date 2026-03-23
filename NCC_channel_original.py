"""
Action-on-rho prototype for the original channel NCC, keeping only the
leading s=2,3 compensated terms.
"""

import argparse
import math
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


def trace_norm(matrix: np.ndarray) -> float:
    """Return the trace norm (Schatten-1 norm) of a matrix."""
    return float(np.sum(np.linalg.svd(matrix, compute_uv=False)))


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


def build_static_data(n: int, j: float, h: float) -> dict:
    """Precompute r-independent operator data for original channel NCC."""
    a_mat, b_mat = build_periodic_ab(n, j, h)
    dim = 2**n
    identity = np.eye(dim, dtype=complex)

    f2 = commutator(b_mat, a_mat)
    f3 = 1j * (
        2 * commutator(b_mat, commutator(a_mat, b_mat))
        + commutator(a_mat, commutator(a_mat, b_mat))
    )
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

    pair_data = paired_channel_parameters(eta_pair_sum)
    pair_scale = pair_data["pair_scale"]

    raw_weights = {
        # Outer order sampling follows eta_s / eta_pair_sum times the common pair_scale.
        2: pair_scale * eta2 / eta_pair_sum,
        3: pair_scale * eta3 / eta_pair_sum,
    }
    raw_total = float(sum(raw_weights.values()))
    p_order = np.array([raw_weights[2] / raw_total, raw_weights[3] / raw_total], dtype=float)

    pair_component_data = {}
    pair_component_probs = {}
    for order, coeffs, labels, l1_norm in (
        (2, static["f2_coeffs"], static["f2_labels"], static["f2_l1"]),
        (3, static["f3_coeffs"], static["f3_labels"], static["f3_l1"]),
    ):
        probs = np.abs(coeffs) / l1_norm
        components = []
        for prob, coeff, label in zip(probs, coeffs, labels):
            phase = coeff / (1j * abs(coeff))
            w_mat = phase * cached_pauli_matrix_from_label(label)
            hermitian_err = float(np.linalg.norm(w_mat - w_mat.conj().T, ord="fro"))
            if hermitian_err > 1e-10:
                raise ValueError(f"sampled W is not Hermitian (herm_err={hermitian_err:.3e})")
            paired_unitary = pair_data["cos_theta"] * identity + 1j * pair_data["sin_theta"] * w_mat
            components.append(
                {
                    "prob": float(prob),
                    "W_mat": w_mat,
                    "paired_unitary": paired_unitary,
                }
            )
        pair_component_data[order] = components
        pair_component_probs[order] = np.array([component["prob"] for component in components], dtype=float)

    def apply_uncompensated_single_step(rho: np.ndarray) -> np.ndarray:
        s1 = expm(-1j * b_mat * t) @ expm(-1j * a_mat * t)
        return apply_unitary_channel(s1, rho)

    s1 = expm(-1j * b_mat * t) @ expm(-1j * a_mat * t)
    step_exact = expm(-1j * h_total * t)
    u_exact = expm(-1j * h_total * t_total)
    v_exact = step_exact @ expm(1j * a_mat * t) @ expm(1j * b_mat * t)

    def apply_pair_component_expectation(component: dict, rho: np.ndarray) -> np.ndarray:
        return (rho + eta_pair_sum * apply_ad_commutator(1j * component["W_mat"], rho)) / pair_scale

    def apply_tilde_V_taylor(rho: np.ndarray) -> np.ndarray:
        return rho + (t**2 / 2) * apply_ad_commutator(static["F2"], rho) + (t**3 / 6) * apply_ad_commutator(static["F3"], rho)

    def apply_tilde_V_compensation(rho: np.ndarray) -> np.ndarray:
        out = np.zeros_like(rho)
        for order in (2, 3):
            for component in pair_component_data[order]:
                # raw_weight_s times the normalized expectation of the paired sampler
                # reconstructs the target I + (t^2/2)F_2 + (t^3/6)F_3 action.
                out += raw_weights[order] * component["prob"] * apply_pair_component_expectation(component, rho)
        return out

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
        return apply_unitary_channel(u_exact, rho)

    validation_error = basis_action_distance(apply_tilde_V_compensation, apply_tilde_V_taylor, dim)
    if validation_error > validation_tol:
        raise ValueError(f"channel tilde_V mismatch between compensation expectation and Taylor action: {validation_error:.3e}")

    deterministic_bias = basis_action_distance(apply_compensated_total, apply_exact_total, dim)
    uncompensated_total_error = basis_action_distance(apply_uncompensated_total, apply_exact_total, dim)
    single_step_error_before = basis_action_distance(apply_uncompensated_single_step, apply_exact_single_step, dim)
    single_step_expectation_bias = basis_action_distance(apply_compensated_single_step, apply_exact_single_step, dim)

    return {
        "t": t,
        "S1": s1,
        "step_exact": step_exact,
        "U_exact": u_exact,
        "V_exact": v_exact,
        "eta": {2: eta2, 3: eta3},
        "eta_pair_sum": eta_pair_sum,
        "raw_weights": raw_weights,
        "raw_total": raw_total,
        "p_order": p_order,
        "pair_component_data": pair_component_data,
        "pair_component_probs": pair_component_probs,
        "pair_orders": (2, 3),
        "validation_error": validation_error,
        "deterministic_bias": deterministic_bias,
        "uncompensated_total_error": uncompensated_total_error,
        "single_step_error_before": single_step_error_before,
        "single_step_expectation_bias": single_step_expectation_bias,
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
) -> np.ndarray:
    """Sample one original compensated remainder channel and apply it to rho."""
    del static
    components = evolution_data["pair_component_data"][order]
    probs = evolution_data["pair_component_probs"][order]
    idx = int(rng.choice(len(components), p=probs))
    component = components[idx]
    if rng.random() < evolution_data["unitary_branch_prob"]:
        # Positive channel branch: U_theta rho U_theta^\dagger.
        out = apply_unitary_channel(component["paired_unitary"], rho)
    else:
        # Negative branch: - P rho P.
        out = -(component["W_mat"] @ rho @ component["W_mat"])
    return evolution_data["raw_total"] * out


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

    rho0 = zero_density_matrix(n)
    rho_single_exact = evolution_data["apply_exact_single_step"](rho0)
    rho_single_before = evolution_data["apply_uncompensated_single_step"](rho0)
    rho_single_deterministic = evolution_data["apply_compensated_single_step"](rho0)

    rho_total_exact = evolution_data["apply_exact_total"](rho0)
    rho_total_before = evolution_data["apply_uncompensated_total"](rho0)
    rho_total_deterministic = evolution_data["apply_compensated_total"](rho0)

    print("action-on-rho original channel prototype:", True)
    print("N:", n, "r:", r)
    print("time step:", evolution_data["t"])
    print("eta2, eta3:", evolution_data["eta"][2], evolution_data["eta"][3])
    print("eta_pair_sum:", evolution_data["eta_pair_sum"])
    print("pair theta:", evolution_data["theta"])
    print("pair scale:", evolution_data["pair_scale"])
    print("raw weights:", evolution_data["raw_weights"])
    print("debug channel validation (basis-level):", evolution_data["validation_error"])

    rng = np.random.default_rng(seed=args.seed)

    def single_step_channel_sampling(num_trials):
        rho_list = [] if args.save_trials_list else None
        average = np.zeros_like(rho0)
        for _ in tqdm(range(num_trials), desc="single-step channel trials"):
            rho_after_trotter = apply_unitary_channel(evolution_data["S1"], rho0)
            order = int(rng.choice((2, 3), p=evolution_data["p_order"]))
            sample = sample_channel_then_compensate(rng, static, evolution_data, order, rho_after_trotter)
            average += sample
            if rho_list is not None:
                rho_list.append(sample)
        average /= num_trials
        return rho_list, average

    rho_single_list, rho_single_avg = single_step_channel_sampling(trials)
    single_step_sample_error = trace_norm(rho_single_avg - rho_single_exact)
    single_step_fluctuation = trace_norm(rho_single_avg - rho_single_deterministic)
    single_step_error_before_state = trace_norm(rho_single_before - rho_single_exact)
    single_step_expectation_bias_state = trace_norm(rho_single_deterministic - rho_single_exact)

    print("single-step trace distance before:", single_step_error_before_state)
    print("single-step trace distance expectation bias:", single_step_expectation_bias_state)
    print("single-step trace distance sample fluctuation:", single_step_fluctuation)
    print("single-step trace distance after compensation:", single_step_sample_error)
    print("debug single-step basis error before:", evolution_data["single_step_error_before"])
    print("debug single-step basis expectation bias:", evolution_data["single_step_expectation_bias"])

    def multi_step_channel_sampling(num_trials):
        rho_list = [] if args.save_trials_list else None
        average = np.zeros_like(rho0)
        for _ in tqdm(range(num_trials), desc="multi-step channel trials"):
            rho = rho0.copy()
            for _ in range(r):
                rho_after_trotter = apply_unitary_channel(evolution_data["S1"], rho)
                order = int(rng.choice((2, 3), p=evolution_data["p_order"]))
                rho = sample_channel_then_compensate(rng, static, evolution_data, order, rho_after_trotter)
            average += rho
            if rho_list is not None:
                rho_list.append(rho)
        average /= num_trials
        return rho_list, average

    rho_total_list, rho_total_avg = multi_step_channel_sampling(trials)
    total_sample_error = trace_norm(rho_total_avg - rho_total_exact)
    total_sample_fluctuation = trace_norm(rho_total_avg - rho_total_deterministic)
    total_error_before_state = trace_norm(rho_total_before - rho_total_exact)
    total_expectation_bias_state = trace_norm(rho_total_deterministic - rho_total_exact)

    print("total trace distance before:", total_error_before_state)
    print("total trace distance expectation bias:", total_expectation_bias_state)
    print("total trace distance sample fluctuation:", total_sample_fluctuation)
    print("total trace distance after compensation:", total_sample_error)
    print("debug total basis error before:", evolution_data["uncompensated_total_error"])
    print("debug total basis expectation bias:", evolution_data["deterministic_bias"])

    data_dir = Path("data/no_search")
    data_dir.mkdir(parents=True, exist_ok=True)
    out = data_dir / f"NCC_channel_original_trials{trials}_N{args.N}_r{args.r}.npz"
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
