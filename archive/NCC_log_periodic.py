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
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=6, help="number of spins")
    parser.add_argument("--J", type=float, default=1.0, help="interaction strength")
    parser.add_argument("--h", type=float, default=1.0, help="transverse field strength")
    parser.add_argument("--T", type=float, default=1.0, help="total evolution time")
    parser.add_argument("--r", type=int, default=20, help="Trotter segments")
    parser.add_argument("--trials", type=int, default=2000, help="Monte Carlo trials")
    parser.add_argument("--epsilon", type=float, default=0.01, help="target precision parameter")
    parser.add_argument(
        "--s0",
        type=int,
        default=0,
        help="max compensated order, 0 means ceil(log(4/epsilon))",
    )
    parser.add_argument(
        "--uniform_terms",
        action="store_true",
        help="uniformly sample nonzero Pauli terms in each commutator order",
    )
    return parser.parse_args()


def pauli_basis(num_qubits):
    single = [I, X, Y, Z]
    basis = [np.array([[1]], dtype=complex)]
    for _ in range(num_qubits):
        nxt = []
        for left in basis:
            for right in single:
                nxt.append(np.kron(left, right))
        basis = nxt
    return basis


def ad_power(op, target, power):
    out = target
    for _ in range(power):
        out = commutator(op, out)
    return out


def build_c_s_first_order(a_mat, b_mat, s):
    """Eq. (123) in PRX for K=1:
    C_s = (-i)^(s+1) * sum_{m+n=s} binom(s,m) ad_B^m ad_A^n B
    """
    total = np.zeros_like(a_mat, dtype=complex)
    for m in range(s + 1):
        n = s - m
        term = ad_power(b_mat, ad_power(a_mat, b_mat, n), m)
        total += math.comb(s, m) * term
    return ((-1j) ** (s + 1)) * total


def antihermitian_pauli_sampler(matrix, basis, tol=1e-10):
    """For anti-Hermitian matrix C = i * sum_j a_j P_j (a_j real)."""
    scale = int(np.sqrt(matrix.shape[0]))
    terms = []
    weights = []
    for p in basis:
        coeff = np.trace(p.conj().T @ matrix) / (2**scale)
        a = -1j * coeff
        if abs(a) <= tol:
            continue
        if abs(np.imag(a)) > 1e-7:
            raise ValueError("non-real Pauli weight found for anti-Hermitian matrix")
        a_real = float(np.real(a))
        if abs(a_real) <= tol:
            continue
        terms.append((1.0 if a_real > 0 else -1.0, p))
        weights.append(abs(a_real))
    if not weights:
        raise ValueError("empty decomposition")
    weights = np.array(weights, dtype=float)
    return terms, weights / np.sum(weights), float(np.sum(weights))


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

    s1 = expm(-1j * b_mat * t) @ expm(-1j * a_mat * t)
    v_exact = expm(-1j * (a_mat + b_mat) * t) @ expm(1j * a_mat * t) @ expm(1j * b_mat * t)
    evolution_exact = expm(-1j * (a_mat + b_mat) * t_total)

    k = 2
    g = 2 * (j + h)
    a_max = 1.0
    kappa = 1
    q0 = int(np.ceil(np.log(4 * n / epsilon)))
    s0 = int(np.ceil(np.log(4 / epsilon))) if args.s0 <= 0 else args.s0
    s0 = max(3, s0)
    lambda_comm = 4 * (a_max * kappa + 1) * q0 * k * g * (2 * n) ** (1 / 2)
    print("q0:", q0, "s0:", s0, "lambda_comm:", lambda_comm)

    basis = pauli_basis(n)
    cs_data = []
    eta = {}
    for s in range(1, s0):
        c_s = build_c_s_first_order(a_mat, b_mat, s)
        terms, probs_weighted, l1_norm = antihermitian_pauli_sampler(c_s, basis)
        if args.uniform_terms:
            probs = np.ones(len(terms), dtype=float) / len(terms)
        else:
            probs = probs_weighted
        cs_data.append((s + 1, terms, probs, l1_norm, c_s))
        eta[s + 1] = l1_norm * (t ** (s + 1)) / math.factorial(s + 1)

    eta_sum = sum(eta.values())
    p_order = np.array([eta[s] for s in range(2, s0 + 1)], dtype=float)
    p_order /= np.sum(p_order)
    theta = np.arctan(eta_sum)
    print("eta_sum:", eta_sum, "theta:", theta)
    print("orders:", {s: eta[s] for s in range(2, s0 + 1)})

    order_to_data = {order: (terms, probs, l1_norm, c_s) for order, terms, probs, l1_norm, c_s in cs_data}

    def sample_w(rng, order):
        terms, probs, _, _ = order_to_data[order]
        idx = int(rng.choice(len(terms), p=probs))
        sign, pauli = terms[idx]
        return sign * pauli

    rng = np.random.default_rng(seed=7)
    s_orders = list(range(2, s0 + 1))

    v_list = []
    for _ in tqdm(range(trials), desc="single step trials"):
        order = int(rng.choice(s_orders, p=p_order))
        w = sample_w(rng, order)
        v_list.append(expm(1j * theta * w))
    v_average = sum(v_list) / trials

    evolution_list = []
    for _ in tqdm(range(trials), desc="multi step trials"):
        evo = np.eye(2**n, dtype=complex)
        for _ in range(r):
            order = int(rng.choice(s_orders, p=p_order))
            w = sample_w(rng, order)
            evo = expm(1j * theta * w) @ s1 @ evo
        evolution_list.append(evo)
    evolution_average = sum(evolution_list) / trials

    single_before = np.linalg.norm(s1 - expm(-1j * (a_mat + b_mat) * t), 2)
    single_after = np.linalg.norm(v_average - v_exact, 2)
    total_before = np.linalg.norm(np.linalg.matrix_power(s1, r) - evolution_exact, 2)
    total_after = np.linalg.norm(evolution_average - evolution_exact, 2)

    print("single error before:", single_before)
    print("single error after:", single_after)
    print("total error before:", total_before)
    print("total error after:", total_after)

    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    mode = "uniform" if args.uniform_terms else "weighted"
    out = data_dir / f"results_log_periodic_{mode}_trials{trials}_s0{s0}.npz"
    np.savez(
        out,
        A=a_mat,
        B=b_mat,
        S=s1,
        V_exact=v_exact,
        V_average=v_average,
        evolution_average=evolution_average,
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
        q0=q0,
        s0=s0,
        theta=theta,
        eta_sum=eta_sum,
        uniform_terms=args.uniform_terms,
    )
    print("saving results to:", out)


if __name__ == "__main__":
    main()
