"""
follow pseudocode
different from operator version(NCC_log.py)
the Phi notation as expectation contains x^q.
"""

import argparse
import math
from functools import lru_cache

import numpy as np
from scipy.linalg import expm
from tqdm import tqdm

from NCC_channel_log import (
    apply_channel_term,
    build_static_data as build_channel_static_data,
    build_tilde_V as build_channel_tilde_V,
    trace_norm,
)
from Pauli_Hamiltonian_BCH import build_periodic_ab, cached_pauli_matrix_from_label, phi_term, tilde_F_term, commutator


def parse_args():
    parser = argparse.ArgumentParser(description="Prototype implementation of the three rejection-sampling pseudocodes.")
    parser.add_argument("--N", type=int, default=2, help="number of spins")
    parser.add_argument("--J", type=float, default=1.0, help="interaction strength")
    parser.add_argument("--h", type=float, default=1.0, help="field strength")
    parser.add_argument("--T", type=float, default=0.2, help="total evolution time")
    parser.add_argument("--r", type=int, default=4, help="number of Trotter segments")
    parser.add_argument("--trials", type=int, default=200, help="Monte Carlo trials")
    parser.add_argument("--seed", type=int, default=7, help="RNG seed")
    parser.add_argument("--epsilon", type=float, default=0.1, help="target precision used for q0/s0 defaults")
    parser.add_argument("--q0", type=int, default=0, help="BCH truncation order")
    parser.add_argument("--s0", type=int, default=0, help="compensation truncation order")
    parser.add_argument("--diagnostic_trials", type=int, default=50000, help="trials used for acceptance-rate diagnostics")
    parser.add_argument(
        "--initial_state",
        choices=("plus_zero", "zero"),
        default="plus_zero",
        help="state used for the single-step sanity check",
    )
    return parser.parse_args()


def support_from_label(label):
    return frozenset(index for index, entry in enumerate(label) if entry != 0)


def initial_density_matrix(num_qubits: int, kind: str) -> np.ndarray:
    """Return a small-state sanity-check density matrix."""
    dim = 2**num_qubits
    psi = np.zeros(dim, dtype=complex)
    if kind == "zero":
        psi[0] = 1.0
    elif kind == "plus_zero":
        psi[0] = 1.0 / math.sqrt(2.0)
        psi[1] = 1.0 / math.sqrt(2.0)
    else:
        raise ValueError(f"unsupported initial_state={kind}")
    return np.outer(psi, psi.conj())


@lru_cache(maxsize=None)
def heisenberg_ab_term_specs(n: int, coupling_j: float, field_h: float):
    """Return local Pauli-term specifications for the periodic A/B partition."""
    terms_a = []
    terms_b = []

    def add_two_body(target, site, pauli_label):
        label = [0] * n
        label[site] = pauli_label
        label[(site + 1) % n] = pauli_label
        target.append((coupling_j, tuple(label)))

    def add_field(target, site):
        label = [0] * n
        label[site] = 3
        target.append((field_h, tuple(label)))

    for site in range(0, n, 2):
        add_two_body(terms_a, site, 1)
        add_two_body(terms_a, site, 2)
        add_two_body(terms_a, site, 3)
        add_field(terms_a, site)

    for site in range(1, n, 2):
        add_two_body(terms_b, site, 1)
        add_two_body(terms_b, site, 2)
        add_two_body(terms_b, site, 3)
        add_field(terms_b, site)

    return tuple(terms_a), tuple(terms_b)


def heisenberg_extensiveness_g(coupling_j: float, field_h: float) -> float:
    """Physical g_0 for the 1D Heisenberg model under the user's convention."""
    return 6.0 * coupling_j + field_h


@lru_cache(maxsize=None)
def build_static(n: int, coupling_j: float, field_h: float):
    """Build x-independent BCH input data for repeated rejection sampling runs."""
    terms_a, terms_b = heisenberg_ab_term_specs(n, coupling_j, field_h)

    def to_term_paulis(specs, term_index, negative=False):
        paulis = []
        for gamma, (alpha, label) in enumerate(specs):
            pauli = cached_pauli_matrix_from_label(label)
            if negative:
                pauli = -pauli
            paulis.append(
                {
                    "alpha_0": float(alpha),
                    "matrix": pauli,
                    "support": support_from_label(label),
                    "term": term_index,
                    "gamma": gamma,
                }
            )
        return tuple(paulis)

    static = (
        to_term_paulis(terms_a + terms_b, 1),  # term 1
        to_term_paulis(terms_a, 2, negative=True),  # term 2
        to_term_paulis(terms_b, 3, negative=True),  # term 3
    )
    return static


def build_tilde_H_v(static, x: float):
    """Build the scaled BCH-input Hamiltonians {x \\tilde H_v} from cached static data."""
    tilde_H_v = []
    for term_family in static:
        tilde_H_v.append(
            [
                {
                    "alpha": float(x * pauli_term["alpha_0"]),
                    "matrix": pauli_term["matrix"],
                    "support": pauli_term["support"],
                    "term": pauli_term["term"],
                    "gamma": pauli_term["gamma"],
                }
                for pauli_term in term_family
            ]
        )
    return tilde_H_v


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


def sigma_coefficient(perm):
    ascents = sum(perm[i] < perm[i + 1] for i in range(len(perm) - 1))
    return ((-1.0) ** ascents) / (len(perm) ** 2 * math.comb(len(perm) - 1, ascents))


def sample_consistent_permutation(rng: np.random.Generator, v_sequence):
    """Uniformly sample sigma from Sigma_v in Eq. (21)."""
    q = len(v_sequence)
    positions_by_term = {}
    for index, term in enumerate(v_sequence):
        positions_by_term.setdefault(term, []).append(index)

    sigma = [None] * q
    offset = 0
    for term in sorted(positions_by_term):
        targets = positions_by_term[term]
        sampled_targets = list(rng.permutation(targets))
        for local_index, target in enumerate(sampled_targets):
            sigma[offset + local_index] = int(target)
        offset += len(targets)
    return tuple(sigma)


def lc_sample(
    rng: np.random.Generator,
    tilde_H_v,
    n: int,
    k_local: int,
    g_v: float,
    q: int,
):
    """Algorithm 2: light-cone commutator sampling."""
    all_terms = [pauli_term for term_family in tilde_H_v for pauli_term in term_family]
    total_weight = float(sum(term["alpha"] for term in all_terms))

    first_probs = np.array([term["alpha"] / total_weight for term in all_terms], dtype=float)
    first_term = all_terms[int(rng.choice(len(all_terms), p=first_probs))]

    if rng.random() > total_weight / (n * g_v):
        return None

    sequence = [first_term["term"]]
    current = 1j * first_term["matrix"]
    active_support = set(first_term["support"])
    candidate_keys = {(term["term"], term["gamma"]) for term in all_terms if active_support & term["support"]}
    term_lookup = {(term["term"], term["gamma"]): term for term in all_terms}

    for depth in range(2, q + 1):
        candidates = [term_lookup[key] for key in sorted(candidate_keys)]
        local_weight = float(sum(term["alpha"] for term in candidates))
        probs = np.array([term["alpha"] / local_weight for term in candidates], dtype=float)
        next_term = candidates[int(rng.choice(len(candidates), p=probs))]

        if rng.random() > local_weight / (depth * k_local * g_v):
            return None

        updated = commutator(1j * next_term["matrix"], current)
        if np.linalg.norm(updated, ord="fro") <= 1e-12:
            return None
        current = updated / 2.0
        sequence.append(next_term["term"])
        active_support |= set(next_term["support"])
        candidate_keys |= {(term["term"], term["gamma"]) for term in all_terms if set(term["support"]) & active_support}

    return tuple(sequence), current


def bch_sample(
    rng: np.random.Generator,
    tilde_H_v,
    n: int,
    k_local: int,
    g_v: float,
    q: int,
):
    """Algorithm 1: sample one BCH term and return a unitary and sign."""
    res = lc_sample(rng, tilde_H_v, n, k_local, g_v, q)
    if res is None:
        return None, 0.0

    v_sequence, iP = res
    sigma = sample_consistent_permutation(rng, v_sequence)
    coeff = sigma_coefficient(sigma)
    eta = float(np.sign(coeff) * ((-1) ** q))

    if rng.random() > abs(coeff):
        return None, 0.0

    P = -1j * iP  # recover P from anti-Hermitian iP
    if rng.random() < 0.5:
        unitary = expm(1j * math.pi / 4 * P)
    else:
        unitary = expm(-1j * math.pi / 4 * P)
        eta *= -1.0
    return unitary, eta


@lru_cache(maxsize=None)
def raw_weight_s_and_q(q0: int, s0: int, K: int, n: int, k_local: int, g_v: float):
    """Unnormalized weights for the full truncated expansion from order K+1 to s0."""
    raw_weights = []
    for s in range(K + 1, s0 + 1):
        max_parts = s // (K + 1)
        for j in range(1, max_parts + 1):
            for q_tuple in iter_compositions(s, j, K + 1, q0):
                weight = 1.0 / math.factorial(j)
                for q in q_tuple:
                    weight *= math.factorial(q) * (2.0 * k_local * g_v) ** q * (1.0 / k_local) * n
                raw_weights.append((s, j, q_tuple, float(weight)))
    return tuple(raw_weights)


def ncc_sample_rejection(
    rng: np.random.Generator,
    tilde_H_v,
    n: int,
    k_local: int,
    g_v: float,
    q0: int,
    s0: int,
    K: int,
):
    """Algorithm 3: sample the higher-order rejection-compensation unitary."""
    raw_weights = raw_weight_s_and_q(q0, s0, K, n, k_local, g_v)
    dim = 2**n
    raw_total = float(sum(weight for _, _, _, weight in raw_weights))
    total_1_norm = 1.0 + raw_total

    if rng.random() < 1.0 / total_1_norm:
        return np.eye(dim, dtype=complex), 1.0, total_1_norm

    weights = np.array([weight for _, _, _, weight in raw_weights], dtype=float)
    probs = weights / raw_total
    chosen_index = int(rng.choice(len(raw_weights), p=probs))
    _, j_parts, q_tuple, _ = raw_weights[chosen_index]

    V = np.eye(dim, dtype=complex)
    eta = 1.0
    for q in q_tuple[:j_parts]:
        sampled_unitary, sampled_sign = bch_sample(rng, tilde_H_v, n, k_local, g_v, q)
        if sampled_unitary is None:
            return np.zeros((dim, dim), dtype=complex), 0.0, total_1_norm
        eta *= sampled_sign
        V = sampled_unitary @ V
    return V, eta, total_1_norm


def V_exact_action_on_rho(n: int, q0: int, s0: int, coupling_j: float, field_h: float, K: int, x: float, rho: np.ndarray):
    """Exact truncated action I + sum_{s=K+1}^{s0} \\widetilde F_{K,s} on rho with x absorbed."""
    A_mat, B_mat = build_periodic_ab(n, coupling_j, field_h)
    A_mat = x * A_mat
    B_mat = x * B_mat
    phi_terms = phi_term(A_mat, B_mat, q0)
    from NCC_channel_log import channel_term_from_operator_pauli, build_sparse_tilde_F_terms
    from Pauli_Hamiltonian_BCH import pauli_decomposition_stream

    phi_channel_terms = {}
    for order in range(K + 1, q0 + 1):
        coeffs, labels, _ = pauli_decomposition_stream(phi_terms[order])
        phi_channel_terms[order] = channel_term_from_operator_pauli(coeffs, labels, n)
    tilde_F_channel_terms = build_sparse_tilde_F_terms(phi_channel_terms, n, K, q0, s0)

    out = rho.copy()
    for order in range(K + 1, s0 + 1):
        out += apply_channel_term(tilde_F_channel_terms[order], rho)
    return out


def single_step_trotter_error(
    args,
    q0: int,
    s0: int,
    k_order: int,
    rho0: np.ndarray,
):
    """Return the standard before/after single-step errors used elsewhere."""
    static = build_channel_static_data(
        n=args.N,
        q0=q0,
        s0=s0,
        epsilon=args.epsilon,
        j=args.J,
        h=args.h,
        K=k_order,
        max_dense_qubits=max(3, args.N),
    )
    evolution_data = build_channel_tilde_V(static, args.T, args.r)
    rho_single_before = evolution_data["apply_uncompensated_single_step"](rho0)
    rho_single_exact = evolution_data["apply_exact_single_step"](rho0)
    rho_single_compensated = evolution_data["apply_compensated_single_step"](rho0)
    return (
        trace_norm(rho_single_before - rho_single_exact),
        trace_norm(rho_single_compensated - rho_single_exact),
        evolution_data,
    )


def estimate_rejection_diagnostics(
    n: int,
    tilde_H_v,
    k_local: int,
    g_v: float,
    q0: int,
    s0: int,
    K: int,
    diagnostic_trials: int,
):
    """Estimate BCHSample acceptance rates and full-sampler nonzero rate."""
    diagnostics = {}
    for q in range(K + 1, q0 + 1):
        rng = np.random.default_rng(1000 + q)
        accepted = 0
        for _ in range(diagnostic_trials):
            unitary, eta = bch_sample(rng, tilde_H_v, n, k_local, g_v, q)
            if unitary is not None:
                accepted += 1
        diagnostics[q] = accepted / diagnostic_trials

    rng = np.random.default_rng(2026)
    identity = 0
    nonzero = 0
    identity_matrix = np.eye(2**n, dtype=complex)
    for _ in range(diagnostic_trials):
        unitary, eta, _ = ncc_sample_rejection(
            rng,
            tilde_H_v,
            n,
            k_local,
            g_v,
            q0,
            s0,
            K,
        )
        if eta == 0.0:
            continue
        if np.allclose(unitary, identity_matrix):
            identity += 1
        else:
            nonzero += 1
    nonzero_rate = nonzero / diagnostic_trials
    estimated_trials_for_ten_nonzero = math.inf if nonzero_rate == 0.0 else math.ceil(10.0 / nonzero_rate)
    return {
        "bch_acceptance_rates": diagnostics,
        "identity_rate": identity / diagnostic_trials,
        "nonzero_rate": nonzero_rate,
        "estimated_trials_for_ten_nonzero": estimated_trials_for_ten_nonzero,
    }


def main():
    args = parse_args()
    K = 1
    k_local = 2
    q0 = int(np.ceil(np.log(2 * args.N / args.epsilon))) if args.q0 <= 0 else args.q0
    q0 = max(3, q0)
    s0 = int(np.ceil(np.log(4 / args.epsilon))) if args.s0 <= 0 else args.s0
    s0 = max(3, s0)
    x = args.T / args.r
    g_0 = heisenberg_extensiveness_g(args.J, args.h)
    amax_kappa_plus_one = 2.0
    g_v = amax_kappa_plus_one * g_0 * x

    static = build_static(args.N, args.J, args.h)
    tilde_H_v = build_tilde_H_v(static, x)
    rho0 = initial_density_matrix(args.N, args.initial_state)
    V_exact = V_exact_action_on_rho(args.N, q0, s0, args.J, args.h, K, x, rho0)
    sample_error_before, expectation_bias, evolution_data = single_step_trotter_error(args, q0, s0, K, rho0)
    rho_before = evolution_data["apply_uncompensated_single_step"](rho0)
    rho_exact = evolution_data["apply_exact_single_step"](rho0)
    diagnostics = estimate_rejection_diagnostics(
        n=args.N,
        tilde_H_v=tilde_H_v,
        k_local=k_local,
        g_v=g_v,
        q0=q0,
        s0=s0,
        K=K,
        diagnostic_trials=args.diagnostic_trials,
    )

    rng = np.random.default_rng(args.seed)
    average = np.zeros_like(rho0)
    average_after = np.zeros_like(rho0)
    total_1_norm = None
    zero_count = 0
    for _ in tqdm(range(args.trials), desc="rejection pseudocode trials"):
        unitary, eta, total_1_norm = ncc_sample_rejection(
            rng,
            tilde_H_v,
            args.N,
            k_local,
            g_v,
            q0,
            s0,
            K,
        )
        if eta == 0.0:
            zero_count += 1
        average += total_1_norm * eta * (unitary @ rho0 @ unitary.conj().T)
        average_after += total_1_norm * eta * (unitary @ rho_before @ unitary.conj().T)
    average /= args.trials
    average_after /= args.trials
    sample_fluctuation = trace_norm(average - V_exact)
    sample_error_after = trace_norm(average_after - rho_exact)
    zero_probability = zero_count / args.trials

    print("Prototype rejection pseudocode implementation:", True)
    print("N:", args.N, "q0:", q0, "s0:", s0, "r:", args.r, "x:", x)
    print("initial_state:", args.initial_state)
    print("g_0:", g_0)
    print("g_v:", g_v)
    print("tilde_H_v:", len(tilde_H_v), "truncated rejection starts at s =", K + 1)
    print("sample_error_before:", sample_error_before)
    print("sample_error_after:", sample_error_after)
    print("expectation_bias:", expectation_bias)
    print("total_1_norm:", total_1_norm)
    print("sample_fluctuations:", sample_fluctuation)
    print("zero_probability:", zero_probability)
    print("bch_acceptance_rates:", diagnostics["bch_acceptance_rates"])
    print("identity_rate:", diagnostics["identity_rate"])
    print("nonzero_rate:", diagnostics["nonzero_rate"])
    print("estimated_trials_for_10_nonzero_samples:", diagnostics["estimated_trials_for_ten_nonzero"])


if __name__ == "__main__":
    main()
