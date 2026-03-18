"""follow pseudocode"""

import argparse
import math
from functools import lru_cache

import numpy as np
from scipy.linalg import expm
from tqdm import tqdm

from NCC_channel_normal import (
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
        "--disable_outer_importance",
        action="store_true",
        help="disable importance-sampled outer distribution over (s, j, q_tuple)",
    )
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

    return terms_a, terms_b


def heisenberg_extensiveness_g(coupling_j: float, field_h: float) -> float:
    """Physical g for the 1D Heisenberg model under the user's convention."""
    return 6.0 * coupling_j + field_h


def build_rejection_layers(n: int, coupling_j: float, field_h: float):
    """Build the three BCH-input Hamiltonians H1=H, H2=-A, H3=-B for V = U S1^†."""
    terms_a, terms_b = heisenberg_ab_term_specs(n, coupling_j, field_h)
    terms_total = [(alpha, cached_pauli_matrix_from_label(label), support_from_label(label)) for alpha, label in terms_a + terms_b]

    def to_layer_records(specs, layer_index, negative=False):
        records = []
        for gamma, (alpha, label) in enumerate(specs):
            pauli = cached_pauli_matrix_from_label(label)
            if negative:
                pauli = -pauli
            records.append(
                {
                    "alpha": float(alpha),
                    "matrix": pauli,
                    "support": support_from_label(label),
                    "layer": layer_index,
                    "gamma": gamma,
                }
            )
        return records

    layers = [
        [
            {
                "alpha": alpha,
                "matrix": pauli,
                "support": support,
                "layer": 1,
                "gamma": gamma,
            }
            for gamma, (alpha, pauli, support) in enumerate(terms_total)
        ],
        to_layer_records(terms_a, 2, negative=True),
        [
            {
                "alpha": alpha,
                "matrix": -cached_pauli_matrix_from_label(label),
                "support": support_from_label(label),
                "layer": 3,
                "gamma": gamma,
            }
            for gamma, (alpha, label) in enumerate(terms_b)
        ],
    ]
    return layers


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
    positions_by_layer = {}
    for index, layer in enumerate(v_sequence):
        positions_by_layer.setdefault(layer, []).append(index)

    sigma = [None] * q
    offset = 0
    for layer in sorted(positions_by_layer):
        targets = positions_by_layer[layer]
        sampled_targets = list(rng.permutation(targets))
        for local_index, target in enumerate(sampled_targets):
            sigma[offset + local_index] = int(target)
        offset += len(targets)
    return tuple(sigma)


def lc_sample(
    rng: np.random.Generator,
    layers,
    n: int,
    k_local: int,
    g: float,
    amax_kappa_plus_one: float,
    q: int,
):
    """Algorithm 2: light-cone commutator sampling."""
    all_terms = [term for layer in layers for term in layer]
    total_weight = float(sum(term["alpha"] for term in all_terms))

    first_probs = np.array([term["alpha"] / total_weight for term in all_terms], dtype=float)
    first_term = all_terms[int(rng.choice(len(all_terms), p=first_probs))]

    # The PDF's step 3 and correctness text disagree; we follow the correctness proof.
    if rng.random() > total_weight / (n * g):
        return None

    sequence = [first_term["layer"]]
    current = 1j * first_term["matrix"]
    active_support = set(first_term["support"])
    candidate_keys = {(term["layer"], term["gamma"]) for term in all_terms if active_support & term["support"]}
    term_lookup = {(term["layer"], term["gamma"]): term for term in all_terms}

    for depth in range(2, q + 1):
        candidates = [term_lookup[key] for key in sorted(candidate_keys)]
        local_weight = float(sum(term["alpha"] for term in candidates))
        probs = np.array([term["alpha"] / local_weight for term in candidates], dtype=float)
        next_term = candidates[int(rng.choice(len(candidates), p=probs))]

        if rng.random() > local_weight / (depth * k_local * amax_kappa_plus_one * g):
            return None

        updated = commutator(1j * next_term["matrix"], current)
        if np.linalg.norm(updated, ord="fro") <= 1e-12:
            return None
        current = updated / 2.0
        sequence.append(next_term["layer"])
        active_support |= set(next_term["support"])
        candidate_keys |= {(term["layer"], term["gamma"]) for term in all_terms if set(term["support"]) & active_support}

    return tuple(sequence), current


def bch_sample(
    rng: np.random.Generator,
    layers,
    n: int,
    k_local: int,
    g: float,
    amax_kappa_plus_one: float,
    q: int,
):
    """Algorithm 1: sample one BCH term and return a unitary and sign."""
    res = lc_sample(rng, layers, n, k_local, g, amax_kappa_plus_one, q)
    if res is None:
        return None, 0.0

    v_sequence, iP = res
    sigma = sample_consistent_permutation(rng, v_sequence)
    coeff = sigma_coefficient(sigma)
    eta = float(np.sign(coeff) * ((-1) ** q))

    if rng.random() > abs(coeff):
        return None, 0.0

    P = -1j * iP
    if rng.random() < 0.5:
        unitary = expm(1j * math.pi / 4 * P)
    else:
        unitary = expm(-1j * math.pi / 4 * P)
        eta *= -1.0
    return unitary, eta


@lru_cache(maxsize=None)
def rejection_raw_entries(q0: int, s0: int, K: int, n: int, k_local: int, g: float, amax_kappa_plus_one: float, x: float):
    """Unnormalized weights for the full truncated expansion from order K+1 to s0."""
    entries = []
    for s in range(K + 1, s0 + 1):
        max_parts = s // (K + 1)
        for j in range(1, max_parts + 1):
            for q_tuple in iter_compositions(s, j, K + 1, q0):
                weight = 1.0 / math.factorial(j)
                for q in q_tuple:
                    weight *= 2.0 * math.factorial(q) * (2.0 * amax_kappa_plus_one * k_local * g) ** (q - 1) * n * g * (x**q)
                entries.append((s, j, q_tuple, float(weight)))
    return tuple(entries)


def ncc_sample_rejection(
    rng: np.random.Generator,
    layers,
    n: int,
    k_local: int,
    g: float,
    q0: int,
    s0: int,
    K: int,
    x: float,
    amax_kappa_plus_one: float,
    outer_probs=None,
):
    """Algorithm 3: sample the higher-order rejection-compensation unitary."""
    entries = rejection_raw_entries(q0, s0, K, n, k_local, g, amax_kappa_plus_one, x)
    dim = 2**n
    raw_total = float(sum(weight for _, _, _, weight in entries))
    normalization = 1.0 + raw_total

    if rng.random() < 1.0 / normalization:
        return np.eye(dim, dtype=complex), 1.0, normalization

    weights = np.array([weight for _, _, _, weight in entries], dtype=float)
    if outer_probs is None:
        probs = weights / raw_total
    else:
        probs = outer_probs
    chosen_index = int(rng.choice(len(entries), p=probs))
    _, j_parts, q_tuple, chosen_weight = entries[chosen_index]
    branch_scale = chosen_weight / (raw_total * probs[chosen_index])

    V = np.eye(dim, dtype=complex)
    eta = branch_scale
    for q in q_tuple[:j_parts]:
        sampled_unitary, sampled_sign = bch_sample(rng, layers, n, k_local, g, amax_kappa_plus_one, q)
        if sampled_unitary is None:
            return np.zeros((dim, dim), dtype=complex), 0.0, normalization
        eta *= sampled_sign
        V = sampled_unitary @ V
    return V, eta, normalization


def V_exact_action_on_rho(n: int, q0: int, s0: int, coupling_j: float, field_h: float, K: int, x: float, rho: np.ndarray):
    """Exact truncated action I + sum_{s=K+1}^{s0} \\widetilde F_{K,s}(x) on rho."""
    A_mat, B_mat = build_periodic_ab(n, coupling_j, field_h)
    phi_terms = phi_term(A_mat, B_mat, q0)
    from NCC_channel_normal import channel_term_from_operator_pauli, build_sparse_tilde_F_terms
    from Pauli_Hamiltonian_BCH import pauli_decomposition_stream

    phi_channel_terms = {}
    for order in range(K + 1, q0 + 1):
        coeffs, labels, _ = pauli_decomposition_stream(phi_terms[order])
        phi_channel_terms[order] = channel_term_from_operator_pauli(coeffs, labels, n)
    tilde_F_channel_terms = build_sparse_tilde_F_terms(phi_channel_terms, n, K, q0, s0)

    out = rho.copy()
    for order in range(K + 1, s0 + 1):
        out += (x**order) * apply_channel_term(tilde_F_channel_terms[order], rho)
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
    layers,
    k_local: int,
    g: float,
    q0: int,
    s0: int,
    K: int,
    x: float,
    amax_kappa_plus_one: float,
    diagnostic_trials: int,
):
    """Estimate BCHSample acceptance rates and full-sampler nonzero rate."""
    diagnostics = {}
    for q in range(K + 1, q0 + 1):
        rng = np.random.default_rng(1000 + q)
        accepted = 0
        for _ in range(diagnostic_trials):
            unitary, eta = bch_sample(rng, layers, n, k_local, g, amax_kappa_plus_one, q)
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
            layers,
            n,
            k_local,
            g,
            q0,
            s0,
            K,
            x,
            amax_kappa_plus_one,
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


def build_outer_importance_distribution(
    q0: int,
    s0: int,
    K: int,
    n: int,
    k_local: int,
    g: float,
    amax_kappa_plus_one: float,
    x: float,
    bch_acceptance_rates,
    diagnostic_trials: int,
):
    """Proposal over (s, j, q_tuple) biased by estimated BCHSample success rates."""
    entries = rejection_raw_entries(q0, s0, K, n, k_local, g, amax_kappa_plus_one, x)
    fallback_rate = 1.0 / diagnostic_trials
    scores = []
    for _, j_parts, q_tuple, weight in entries:
        estimated_success = 1.0
        for q in q_tuple[:j_parts]:
            estimated_success *= max(bch_acceptance_rates.get(q, 0.0), fallback_rate)
        scores.append(weight * estimated_success)
    total_score = float(sum(scores))
    if total_score <= 0.0:
        return np.array([weight for _, _, _, weight in entries], dtype=float) / sum(weight for _, _, _, weight in entries)
    return np.array(scores, dtype=float) / total_score


def main():
    args = parse_args()
    K = 1
    k_local = 2
    q0 = int(np.ceil(np.log(2 * args.N / args.epsilon))) if args.q0 <= 0 else args.q0
    q0 = max(3, q0)
    s0 = int(np.ceil(np.log(4 / args.epsilon))) if args.s0 <= 0 else args.s0
    s0 = max(3, s0)
    x = args.T / args.r
    g = heisenberg_extensiveness_g(args.J, args.h)

    layers = build_rejection_layers(args.N, args.J, args.h)
    rho0 = initial_density_matrix(args.N, args.initial_state)
    V_exact = V_exact_action_on_rho(args.N, q0, s0, args.J, args.h, K, x, rho0)
    sample_error_before, expectation_bias, _ = single_step_trotter_error(args, q0, s0, K, rho0)
    diagnostics = estimate_rejection_diagnostics(
        n=args.N,
        layers=layers,
        k_local=k_local,
        g=g,
        q0=q0,
        s0=s0,
        K=K,
        x=x,
        amax_kappa_plus_one=2.0,
        diagnostic_trials=args.diagnostic_trials,
    )
    outer_probs = None
    if not args.disable_outer_importance:
        outer_probs = build_outer_importance_distribution(
            q0=q0,
            s0=s0,
            K=K,
            n=args.N,
            k_local=k_local,
            g=g,
            amax_kappa_plus_one=2.0,
            x=x,
            bch_acceptance_rates=diagnostics["bch_acceptance_rates"],
            diagnostic_trials=args.diagnostic_trials,
        )

    rng = np.random.default_rng(args.seed)
    average = np.zeros_like(rho0)
    normalization = None
    for _ in tqdm(range(args.trials), desc="rejection pseudocode trials"):
        unitary, eta, normalization = ncc_sample_rejection(
            rng,
            layers,
            args.N,
            k_local,
            g,
            q0,
            s0,
            K,
            x,
            amax_kappa_plus_one=2.0,
            outer_probs=outer_probs,
        )
        average += normalization * eta * (unitary @ rho0 @ unitary.conj().T)
    average /= args.trials
    sample_fluctuation = trace_norm(average - V_exact)

    print("Prototype rejection pseudocode implementation:", True)
    print("N:", args.N, "q0:", q0, "s0:", s0, "r:", args.r, "x:", x)
    print("initial_state:", args.initial_state)
    print("g:", g)
    print("layers:", len(layers), "truncated rejection starts at s =", K + 1)
    print("sample_error_before:", sample_error_before)
    print("expectation_bias:", expectation_bias)
    print("normalization:", normalization)
    print("sample_fluctuations:", sample_fluctuation)
    print("bch_acceptance_rates:", diagnostics["bch_acceptance_rates"])
    print("identity_rate:", diagnostics["identity_rate"])
    print("nonzero_rate:", diagnostics["nonzero_rate"])
    print("estimated_trials_for_10_nonzero_samples:", diagnostics["estimated_trials_for_ten_nonzero"])
    print("outer_importance_enabled:", not args.disable_outer_importance)


if __name__ == "__main__":
    main()
