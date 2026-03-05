import argparse
import math
from pathlib import Path

import numpy as np
from scipy.linalg import expm
from tqdm import tqdm


X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
I = np.eye(2, dtype=complex)


def commutator(a, b):
    return a @ b - b @ a


def parse_args():
    parser = argparse.ArgumentParser(
        description="NCC log-precision prototype with periodic-boundary simplification."
    )
    parser.add_argument("--N", type=int, default=6, help="number of spins")
    parser.add_argument("--J", type=float, default=1.0, help="interaction strength")
    parser.add_argument("--h", type=float, default=1.0, help="field strength")
    parser.add_argument("--T", type=float, default=1.0, help="total evolution time")
    parser.add_argument("--r", type=int, default=20, help="number of Trotter segments")
    parser.add_argument("--trials", type=int, default=2000, help="Monte Carlo trials")
    parser.add_argument("--epsilon", type=float, default=0.01, help="target precision")
    parser.add_argument(
        "--s0",
        type=int,
        default=0,
        help="max compensated order (0 => ceil(log(4/epsilon)))",
    )
    parser.add_argument(
        "--sampling",
        choices=["uniform", "weighted"],
        default="uniform",
        help="term sampling inside each commutator order",
    )
    return parser.parse_args()


def pauli_basis(n):
    single = [I, X, Y, Z]
    basis = [np.array([[1]], dtype=complex)]
    for _ in range(n):
        nxt = []
        for left in basis:
            for right in single:
                nxt.append(np.kron(left, right))
        basis = nxt
    return basis


def ad_power(op, target, p):
    out = target
    for _ in range(p):
        out = commutator(op, out)
    return out


def c_s_first_order(a_mat, b_mat, s):
    """PRX Eq. (123) for K=1:
    C_s = (-i)^(s+1) sum_{m+n=s} binom(s,m) ad_B^m ad_A^n B
    """
    total = np.zeros_like(a_mat, dtype=complex)
    for m in range(s + 1):
        n = s - m
        total += math.comb(s, m) * ad_power(b_mat, ad_power(a_mat, b_mat, n), m)
    return ((-1j) ** (s + 1)) * total


def antihermitian_pauli_decomposition(mat, basis, tol=1e-10):
    """Decompose anti-Hermitian mat as i * sum_j a_j P_j (a_j real)."""
    n = int(round(math.log2(mat.shape[0])))
    scale = 2**n
    terms = []
    abs_weights = []
    for p in basis:
        coeff = np.trace(p.conj().T @ mat) / scale
        a = -1j * coeff
        if abs(a) <= tol:
            continue
        if abs(np.imag(a)) > 1e-7:
            raise ValueError("decomposition coefficient has non-negligible imaginary part")
        a_real = float(np.real(a))
        if abs(a_real) <= tol:
            continue
        terms.append((1.0 if a_real > 0 else -1.0, p))
        abs_weights.append(abs(a_real))
    if not abs_weights:
        raise ValueError("empty anti-Hermitian Pauli decomposition")
    abs_weights = np.array(abs_weights, dtype=float)
    probs_weighted = abs_weights / np.sum(abs_weights)
    l1_norm = float(np.sum(abs_weights))
    return terms, probs_weighted, l1_norm


def build_periodic_ab(n, j, h):
    def xx_term(index):
        if index < n - 1:
            return j * np.kron(
                np.eye(2**index, dtype=complex),
                np.kron(np.kron(X, X), np.eye(2 ** (n - index - 2), dtype=complex)),
            )
        return j * np.kron(X, np.kron(np.eye(2 ** (n - 2), dtype=complex), X))

    def z_term(index):
        return h * np.kron(
            np.eye(2**index, dtype=complex),
            np.kron(Z, np.eye(2 ** (n - index - 1), dtype=complex)),
        )

    a_mat = np.zeros((2**n, 2**n), dtype=complex)
    b_mat = np.zeros((2**n, 2**n), dtype=complex)
    for idx in range(0, n, 2):
        a_mat += xx_term(idx) + z_term(idx)
    for idx in range(1, n, 2):
        b_mat += xx_term(idx) + z_term(idx)
    return a_mat, b_mat


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

    # Hamiltonian with periodic boundary condition
    a_mat, b_mat = build_periodic_ab(n, j, h)
    s1 = expm(-1j * b_mat * t) @ expm(-1j * a_mat * t)
    v_exact = expm(-1j * (a_mat + b_mat) * t) @ expm(1j * a_mat * t) @ expm(1j * b_mat * t)
    u_exact = expm(-1j * (a_mat + b_mat) * t_total)

    # q0/s0 and convergence checks from NCC_with_log_precision.pdf
    k_local = 2
    g = 2 * (j + h)
    a_max = 1.0
    kappa = 1
    coeff = a_max * kappa + 1
    q0 = int(np.ceil(np.log(4 * n / epsilon)))
    s0 = int(np.ceil(np.log(4 / epsilon))) if args.s0 <= 0 else args.s0
    s0 = max(3, s0)
    lambda_comm = 4 * coeff * q0 * k_local * g * (2 * n) ** (1 / 2)

    cond_lemma1 = 8 * (math.e**2) * k_local * q0 * coeff * g * t
    cond_lemma3 = (math.e**2) * lambda_comm * t
    print("q0:", q0, "s0:", s0, "lambda_comm:", lambda_comm)
    print("cond(lemma1)<=1:", cond_lemma1)
    print("cond(lemma3)<=1:", cond_lemma3)

    basis = pauli_basis(n)
    order_data = {}
    eta = {}
    for s in range(1, s0):
        c_s = c_s_first_order(a_mat, b_mat, s)
        terms, weighted_probs, l1_norm = antihermitian_pauli_decomposition(c_s, basis)
        if args.sampling == "uniform":
            probs = np.ones(len(terms), dtype=float) / len(terms)
        else:
            probs = weighted_probs
        order = s + 1
        order_data[order] = (terms, probs, l1_norm)
        eta[order] = l1_norm * (t ** order) / math.factorial(order)

    eta_sum = sum(eta.values())
    s_orders = list(range(2, s0 + 1))
    p_order = np.array([eta[s] for s in s_orders], dtype=float)
    p_order = p_order / np.sum(p_order)
    theta = np.arctan(eta_sum)
    print("eta_sum:", eta_sum, "theta:", theta)
    print("eta by order:", {s: eta[s] for s in s_orders})

    rng = np.random.default_rng(seed=7)

    def sample_w(order):
        terms, probs, _ = order_data[order]
        idx = int(rng.choice(len(terms), p=probs))
        sign, pauli = terms[idx]
        return sign * pauli

    v_list = []
    for _ in tqdm(range(trials), desc="single step trials"):
        order = int(rng.choice(s_orders, p=p_order))
        w = sample_w(order)
        v_list.append(expm(1j * theta * w))
    v_avg = sum(v_list) / trials

    evo_list = []
    for _ in tqdm(range(trials), desc="multi step trials"):
        evo = np.eye(2**n, dtype=complex)
        for _ in range(r):
            order = int(rng.choice(s_orders, p=p_order))
            w = sample_w(order)
            evo = expm(1j * theta * w) @ s1 @ evo
        evo_list.append(evo)
    evo_avg = sum(evo_list) / trials

    single_before = np.linalg.norm(s1 - expm(-1j * (a_mat + b_mat) * t), 2)
    single_after = np.linalg.norm(v_avg - v_exact, 2)
    total_before = np.linalg.norm(np.linalg.matrix_power(s1, r) - u_exact, 2)
    total_after = np.linalg.norm(evo_avg - u_exact, 2)
    print("single error before:", single_before)
    print("single error after:", single_after)
    print("total error before:", total_before)
    print("total error after:", total_after)

    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    out = data_dir / f"results_log_{args.sampling}_trials{trials}_s0{s0}.npz"
    np.savez(
        out,
        A=a_mat,
        B=b_mat,
        S=s1,
        V_exact=v_exact,
        V_average=v_avg,
        evolution_average=evo_avg,
        single_step_error_before=single_before,
        single_step_error_after=single_after,
        total_error_before=total_before,
        total_error_after=total_after,
        N=n,
        J=j,
        h=h,
        T=t_total,
        r=r,
        trials=trials,
        epsilon=epsilon,
        q0=q0,
        s0=s0,
        theta=theta,
        eta_sum=eta_sum,
        sampling=args.sampling,
        cond_lemma1=cond_lemma1,
        cond_lemma3=cond_lemma3,
    )
    print("saving results to:", out)


if __name__ == "__main__":
    main()
