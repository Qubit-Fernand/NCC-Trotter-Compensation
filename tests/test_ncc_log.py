import math
import sys
from pathlib import Path

import numpy as np
from scipy.linalg import expm

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from NCC_log import (
    build_periodic_ab,
    commutator,
    pauli_basis,
    pauli_decomposition,
    phi_term,
    phi_term_by_log,
    tilde_F_term,
)


def reference_phi_terms_k1(a_mat, b_mat):
    c1 = commutator(b_mat, a_mat)
    c2 = 1j * (
        2 * commutator(b_mat, commutator(a_mat, b_mat))
        + commutator(a_mat, commutator(a_mat, b_mat))
    )
    return {
        2: c1 / math.factorial(2),
        3: c2 / math.factorial(3),
    }


def truncated_bch_exponential(phi_terms, x, k_order, q0):
    generator = np.zeros_like(next(iter(phi_terms.values())))
    for q in range(k_order + 1, q0 + 1):
        generator += phi_terms[q] * (x**q)
    return expm(generator)


def validate_phi_and_tilde_f(n=4, j=1.0, h=1.0, t=0.05, k_order=1, s0=4):
    a_mat, b_mat = build_periodic_ab(n, j, h)
    phi_terms, _ = phi_term(a_mat, b_mat, s0)
    phi_terms_by_log, _ = phi_term_by_log(a_mat, b_mat, s0, base_step=min(t, 0.02))
    tilde_f_terms = tilde_F_term(phi_terms, k_order, s0, s0)
    ref_phi = reference_phi_terms_k1(a_mat, b_mat)
    basis = pauli_basis(n)

    metrics = {}
    for q in sorted(ref_phi):
        if q > s0:
            continue
        ref_norm = max(np.linalg.norm(ref_phi[q], 2), 1e-12)
        metrics[f"phi_{q}_relative"] = np.linalg.norm(phi_terms[q] - ref_phi[q], 2) / ref_norm
        metrics[f"phi_{q}_vs_log_fit"] = np.linalg.norm(phi_terms[q] - phi_terms_by_log[q], 2) / ref_norm
        metrics[f"tilde_f_{q}_relative"] = np.linalg.norm(tilde_f_terms[q] - ref_phi[q], 2) / ref_norm

    single_phi_max = min(2 * k_order + 1, s0)
    for s in range(k_order + 1, single_phi_max + 1):
        phi_norm = max(np.linalg.norm(phi_terms[s], 2), 1e-12)
        metrics[f"single_phi_regime_{s}"] = np.linalg.norm(tilde_f_terms[s] - phi_terms[s], 2) / phi_norm

    taylor_eval = np.eye(2**n, dtype=complex)
    for s, term in tilde_f_terms.items():
        taylor_eval += term * (t**s)
    truncated_exp_eval = truncated_bch_exponential(phi_terms, t, k_order, s0)
    metrics["taylor_vs_exp_error"] = np.linalg.norm(taylor_eval - truncated_exp_eval, 2)

    eta = {}
    order_data = {}
    for order in range(2, s0 + 1):
        terms, weighted_probs, l1_norm = pauli_decomposition(tilde_f_terms[order], basis)
        order_data[order] = {"kind": "tail", "terms": terms, "probs": weighted_probs, "l1_norm": l1_norm}
        eta[order] = l1_norm * (t**order)
        antiherm_err = np.linalg.norm(tilde_f_terms[order] + tilde_f_terms[order].conj().T, ord="fro")
        if antiherm_err <= 1e-8:
            pair_terms, pair_probs, _ = pauli_decomposition(
                tilde_f_terms[order],
                basis,
                antihermitian=True,
            )
            order_data[order] = {"kind": "pair", "terms": pair_terms, "probs": pair_probs, "l1_norm": l1_norm}

    s_orders = list(range(2, s0 + 1))
    leading_orders = [s for s in s_orders if s <= 2 * k_order + 1]
    tail_orders = [s for s in s_orders if s > 2 * k_order + 1]
    eta_pair_sum = sum(eta[s] for s in leading_orders)
    theta_pair = np.arctan(eta_pair_sum)
    raw_weights = {}
    if eta_pair_sum > 0:
        for s in leading_orders:
            raw_weights[s] = eta[s] / eta_pair_sum
    for s in tail_orders:
        raw_weights[s] = eta[s]

    tilde_v_comp = np.zeros((2**n, 2**n), dtype=complex)
    for order in s_orders:
        data = order_data[order]
        for prob, (phase, pauli) in zip(data["probs"], data["terms"]):
            if data["kind"] == "pair":
                tilde_v_comp += raw_weights[order] * prob * expm(1j * theta_pair * (phase * pauli))
            else:
                tilde_v_comp += raw_weights[order] * prob * (phase * pauli)

    metrics["tilde_v_comp_vs_taylor"] = np.linalg.norm(tilde_v_comp - taylor_eval, 2)
    return metrics


def main():
    metrics = validate_phi_and_tilde_f()
    print(metrics)

    assert metrics["phi_2_relative"] < 1e-8
    assert metrics["phi_2_vs_log_fit"] < 1e-8
    assert metrics["tilde_f_2_relative"] < 1e-8
    assert metrics["phi_3_relative"] < 1e-5
    assert metrics["phi_3_vs_log_fit"] < 1e-5
    assert metrics["tilde_f_3_relative"] < 1e-5
    assert metrics["single_phi_regime_2"] < 1e-12
    assert metrics["single_phi_regime_3"] < 1e-12
    assert metrics["taylor_vs_exp_error"] < 1e-4
    assert metrics["tilde_v_comp_vs_taylor"] < 5e-3


if __name__ == "__main__":
    main()
