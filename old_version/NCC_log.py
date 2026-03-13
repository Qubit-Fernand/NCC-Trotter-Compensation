# 第250行本应该传入q0和s0，错误传了s0两次
# 第249行也应该传入q0
import argparse
import itertools
import math
from functools import lru_cache

import numpy as np
from scipy.linalg import expm, logm
from tqdm import tqdm


X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
I = np.eye(2, dtype=complex)


def commutator(a, b):
    return a @ b - b @ a


def parse_args():
    parser = argparse.ArgumentParser(description="NCC log-precision prototype with periodic-boundary simplification.")
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


def build_periodic_ab(n, j, h):
    """Build A and B matrices for the Hamiltonian H = sum_{i} j X_i X_{i+1} + h Z_i with periodic boundary condition."""

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


def phi_term_by_log(a_mat, b_mat, q_max, base_step=None):
    """Archived numeric extractor: fit Phi_q from log(V(x)) near x=0."""

    def bch_remainder(a_mat, b_mat, x):
        return expm(-1j * (a_mat + b_mat) * x) @ expm(1j * a_mat * x) @ expm(1j * b_mat * x)

    def logm_close_to_identity(unitary_like):
        omega = logm(unitary_like)
        omega = 0.5 * (omega - omega.conj().T)
        return omega

    dim = a_mat.shape[0]
    if base_step is None:
        op_scale = max(
            np.linalg.norm(a_mat, 2),
            np.linalg.norm(b_mat, 2),
            np.linalg.norm(a_mat + b_mat, 2),
            1.0,
        )
        base_step = min(0.02, 0.2 / op_scale)
    base_step = max(float(base_step), 1e-3)

    phi_terms = {}
    for q in range(2, q_max + 1):
        fit_order = q_max - q + 3
        xs = base_step / (2.0 ** np.arange(fit_order))
        samples = []
        for x in xs:
            omega = logm_close_to_identity(bch_remainder(a_mat, b_mat, x))
            residual = omega.copy()
            for lower_q in range(2, q):
                residual -= phi_terms[lower_q] * (x**lower_q)
            samples.append((residual / (x**q)).reshape(-1))
        vand = np.vander(xs, N=fit_order, increasing=True)
        coeffs = np.linalg.solve(vand, np.asarray(samples))
        phi_terms[q] = coeffs[0].reshape(dim, dim)
    return phi_terms, base_step


def phi_term(a_mat, b_mat, q_max, base_step=None):
    """Compute Phi_q directly from the BCH commutator formula in the PDF.

    ``base_step`` is accepted for backward compatibility with older notebook
    code paths that used the archived log-fit extractor.
    """

    def compositions(total, parts):
        if parts == 1:
            yield (total,)
            return
        for first in range(total + 1):
            for rest in compositions(total - first, parts - 1):
                yield (first,) + rest

    def descents(perm):
        return sum(perm[i] > perm[i + 1] for i in range(len(perm) - 1))

    def right_nested_commutator(ops):
        out = ops[-1]
        for op in reversed(ops[:-1]):
            out = commutator(op, out)
        return out

    x_ops = [1j * b_mat, 1j * a_mat, -1j * (a_mat + b_mat)]  # X1, X2, X3
    dim = a_mat.shape[0]
    phi_terms = {}
    for q in range(2, q_max + 1):
        total = np.zeros((dim, dim), dtype=complex)
        perm_data = []
        for perm in itertools.permutations(range(q)):
            d_sigma = descents(perm)
            coeff = ((-1.0) ** d_sigma) / (q**2 * math.comb(q - 1, d_sigma))
            perm_data.append((perm, coeff))

        for p1, p2, p3 in compositions(q, 3):
            prefactor = 1.0 / (math.factorial(p1) * math.factorial(p2) * math.factorial(p3))
            base_seq = [x_ops[2]] * p3 + [x_ops[1]] * p2 + [x_ops[0]] * p1
            for perm, coeff in perm_data:
                reordered = [base_seq[idx] for idx in perm]
                total += prefactor * coeff * right_nested_commutator(reordered)
        phi_terms[q] = total
    return phi_terms, None


def tilde_F_term(phi_terms, k_order, q0, s0):
    """Return matrices C_s with \tilde F_{K,s}(x) = C_s x^s."""

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

    dim = next(iter(phi_terms.values())).shape[0]
    tilde_f_terms = {}
    q_min = k_order + 1
    for s in range(q_min, s0 + 1):
        total = np.zeros((dim, dim), dtype=complex)
        max_parts = s // q_min
        for j in range(1, max_parts + 1):
            for q_tuple in iter_compositions(s, j, q_min, q0):
                product = np.eye(dim, dtype=complex)
                for q in q_tuple:
                    product = product @ phi_terms[q]
                total += product / math.factorial(j)
        tilde_f_terms[s] = total
    return tilde_f_terms


def pauli_decomposition(mat, basis, antihermitian=False, tol=1e-10):
    """Decompose mat in the Pauli basis."""
    n = int(round(math.log2(mat.shape[0])))
    scale = 2**n
    terms = []
    abs_weights = []
    for p in basis:
        coeff = np.trace(p.conj().T @ mat) / scale
        if antihermitian:
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
        else:
            if abs(coeff) <= tol:
                continue
            coeff_abs = abs(coeff)
            terms.append((coeff / coeff_abs, p))
            abs_weights.append(coeff_abs)
    if not abs_weights:
        kind = "anti-Hermitian Pauli" if antihermitian else "Pauli"
        raise ValueError(f"empty {kind} decomposition")
    abs_weights = np.array(abs_weights, dtype=float)
    probs_weighted = abs_weights / np.sum(abs_weights)
    l1_norm = float(np.sum(abs_weights))
    return terms, probs_weighted, l1_norm


def pauli_rotation(pauli, phase, angle, identity):
    """Closed-form exp(i angle * phase * P) for a Hermitian Pauli P."""
    signed_pauli = phase * pauli
    return np.cos(angle) * identity + 1j * np.sin(angle) * signed_pauli


@lru_cache(maxsize=None)
def build_log_static_data(n, epsilon, j=1.0, h=1.0, sampling="weighted", kappa=1, s0=None):
    """Precompute r-independent data for log-NCC evaluation."""
    if s0 is None:
        s0 = max(3, int(np.ceil(np.log(4 / epsilon))))
    else:
        s0 = max(3, int(s0))

    a_mat, b_mat = build_periodic_ab(n, j, h)
    basis = pauli_basis(n)
    phi_terms, _ = phi_term(a_mat, b_mat, s0)
    tilde_f_terms = tilde_F_term(phi_terms, kappa, s0, s0)
    identity = np.eye(2**n, dtype=complex)

    order_data = {}
    phi_l1 = {}
    tilde_f_l1 = {}
    pairable_orders = []
    non_pairable_orders = []
    for order in range(kappa + 1, s0 + 1):
        phi_q = phi_terms[order]
        _, _, phi_q_l1 = pauli_decomposition(phi_q, basis, antihermitian=True)
        phi_l1[order] = phi_q_l1

        terms, weighted_probs, l1_norm = pauli_decomposition(tilde_f_terms[order], basis)
        probs = weighted_probs if sampling == "weighted" else np.ones(len(terms), dtype=float) / len(terms)
        order_data[order] = {"kind": "tail", "terms": terms, "probs": probs, "l1_norm": l1_norm}
        tilde_f_l1[order] = l1_norm

        antiherm_err = np.linalg.norm(
            tilde_f_terms[order] + tilde_f_terms[order].conj().T,
            ord="fro",
        )
        if antiherm_err <= 1e-8:
            pair_terms, pair_probs, _ = pauli_decomposition(
                tilde_f_terms[order],
                basis,
                antihermitian=True,
            )
            pairable_orders.append(order)
            order_data[order] = {
                "kind": "pair",
                "terms": pair_terms,
                "probs": pair_probs if sampling == "weighted" else probs,
                "l1_norm": l1_norm,
            }
        else:
            non_pairable_orders.append((order, float(antiherm_err)))

    return {
        "a_mat": a_mat,
        "b_mat": b_mat,
        "basis": basis,
        "identity": identity,
        "phi_terms": phi_terms,
        "tilde_f_terms": tilde_f_terms,
        "phi_l1": phi_l1,
        "tilde_f_l1": tilde_f_l1,
        "order_data": order_data,
        "s_orders": list(range(kappa + 1, s0 + 1)),
        "pairable_orders": pairable_orders,
        "non_pairable_orders": non_pairable_orders,
        "kappa": kappa,
        "s0": s0,
    }


def build_log_tilde_v(n, t_total, r, epsilon, j=1.0, h=1.0, sampling="weighted", kappa=1, s0=None):
    """Build the compensated single-step expectation operator tilde_V."""
    static = build_log_static_data(n, epsilon, j=j, h=h, sampling=sampling, kappa=kappa, s0=s0)
    a_mat = static["a_mat"]
    b_mat = static["b_mat"]
    identity = static["identity"]
    s_orders = static["s_orders"]
    order_data = static["order_data"]

    t = t_total / r
    s1 = expm(-1j * b_mat * t) @ expm(-1j * a_mat * t)
    u_exact = expm(-1j * (a_mat + b_mat) * t_total)

    eta = {order: order_data[order]["l1_norm"] * (t**order) for order in s_orders}
    leading_orders = [s for s in s_orders if s <= 2 * kappa + 1]
    tail_orders = [s for s in s_orders if s > 2 * kappa + 1]
    eta_pair_sum = sum(eta[s] for s in leading_orders)
    theta_pair = np.arctan(eta_pair_sum)

    raw_weights = {}
    if eta_pair_sum > 0:
        for s in leading_orders:
            raw_weights[s] = eta[s] / eta_pair_sum
    for s in tail_orders:
        raw_weights[s] = eta[s]

    tilde_v = np.zeros_like(identity)
    for order in s_orders:
        data = order_data[order]
        weight = raw_weights[order]
        for prob, (phase, pauli) in zip(data["probs"], data["terms"]):
            if data["kind"] == "pair":
                tilde_v += weight * prob * pauli_rotation(pauli, phase, theta_pair, identity)
            else:
                tilde_v += weight * prob * (phase * pauli)

    return {
        "tilde_v": tilde_v,
        "s1": s1,
        "u_exact": u_exact,
        "eta": eta,
        "eta_pair_sum": eta_pair_sum,
        "theta_pair": theta_pair,
        "raw_weights": raw_weights,
    }


@lru_cache(maxsize=None)
def exact_log_total_error(n, t_total, r, epsilon, j=1.0, h=1.0, sampling="weighted", kappa=1, s0=None):
    """Return ||(tilde_V S_1)^r - U_exact||_2 for the deterministic expectation operator."""
    data = build_log_tilde_v(n, t_total, r, epsilon, j=j, h=h, sampling=sampling, kappa=kappa, s0=s0)
    return np.linalg.norm(np.linalg.matrix_power(data["tilde_v"] @ data["s1"], r) - data["u_exact"], 2)


def find_min_segments_log(n, t_total, epsilon, j=1.0, h=1.0, sampling="weighted", r_max=512, kappa=1, s0=None):
    """Binary search the smallest r with deterministic log-NCC error <= epsilon."""
    low = 1
    high = 1
    err_high = exact_log_total_error(n, t_total, high, epsilon, j=j, h=h, sampling=sampling, kappa=kappa, s0=s0)
    while err_high > epsilon and high < r_max:
        low = high
        high *= 2
        err_high = exact_log_total_error(
            n,
            t_total,
            high,
            epsilon,
            j=j,
            h=h,
            sampling=sampling,
            kappa=kappa,
            s0=s0,
        )
    if err_high > epsilon:
        raise RuntimeError(f"failed to reach epsilon={epsilon} by r={r_max}")

    while low + 1 < high:
        mid = (low + high) // 2
        err_mid = exact_log_total_error(n, t_total, mid, epsilon, j=j, h=h, sampling=sampling, kappa=kappa, s0=s0)
        if err_mid <= epsilon:
            high = mid
            err_high = err_mid
        else:
            low = mid
    return high, err_high


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

    cond_bch_truncation = 8 * math.e * k_local * q0 * coeff * g * t
    cond_finite_s_truncation = math.e * lambda_comm * t
    print("q0:", q0, "s0:", s0, "lambda_comm:", lambda_comm)
    print("Lemma 3 condition 8e(a_max*kappa+1)q0kg t <= 1:", cond_bch_truncation)
    print("Lemma 5 condition e*lambda_comm*t <= 1:", cond_finite_s_truncation)

    static = build_log_static_data(n, epsilon, j=j, h=h, sampling=args.sampling, kappa=kappa, s0=s0)
    step_data = build_log_tilde_v(n, t_total, r, epsilon, j=j, h=h, sampling=args.sampling, kappa=kappa, s0=s0)
    a_mat = static["a_mat"]
    b_mat = static["b_mat"]
    phi_terms = static["phi_terms"]
    tilde_f_terms = static["tilde_f_terms"]
    phi_l1 = static["phi_l1"]
    tilde_f_l1 = static["tilde_f_l1"]
    order_data = static["order_data"]
    s_orders = static["s_orders"]
    pairable_orders = static["pairable_orders"]
    non_pairable_orders = static["non_pairable_orders"]
    s1 = step_data["s1"]
    tilde_v = step_data["tilde_v"]
    u_exact = step_data["u_exact"]
    eta = step_data["eta"]
    eta_pair_sum = step_data["eta_pair_sum"]
    theta_pair = step_data["theta_pair"]
    raw_weights = step_data["raw_weights"]
    v_exact = expm(-1j * (a_mat + b_mat) * t) @ expm(1j * a_mat * t) @ expm(1j * b_mat * t)
    print("Phi extraction method:", "direct BCH commutator formula")
    raw_total = float(sum(raw_weights.values()))
    p_order = np.array([raw_weights[s] / raw_total for s in s_orders], dtype=float)
    print("eta_pair_sum:", eta_pair_sum, "theta_pair:", theta_pair)
    print("eta by order:", {s: eta[s] for s in s_orders})
    print("Phi_q l1:", {s: phi_l1[s] for s in s_orders})
    print("tilde_F Pauli-l1:", {s: tilde_f_l1[s] for s in s_orders})
    print("mixed raw weights:", {s: raw_weights[s] for s in s_orders})
    print("Euler-pairable orders:", pairable_orders)
    if non_pairable_orders:
        print("non-pairable orders (anti-Hermitian defect):", non_pairable_orders)

    rng = np.random.default_rng(seed=7)

    def compensation_unitary(w_mat, angle, atol=1e-10):
        """Build compensation unitary exp(i angle W) for Hermitian Pauli W."""
        hermitian_err = np.linalg.norm(w_mat - w_mat.conj().T, ord="fro")
        if hermitian_err > atol:
            raise ValueError(f"sampled W is not Hermitian (herm_err={hermitian_err:.3e})")
        return pauli_rotation(w_mat, 1.0, angle, static["identity"])

    def sample_component(order):
        data = order_data[order]
        terms = data["terms"]
        probs = data["probs"]
        idx = int(rng.choice(len(terms), p=probs))
        phase, pauli = terms[idx]
        if data["kind"] == "pair":
            return compensation_unitary(phase * pauli, theta_pair)
        return phase * pauli

    """The expectatiion value of the sampled component, two ways to compute"""

    def tilde_V():
        tilde_v_local = np.zeros((2**n, 2**n), dtype=complex)
        for order in s_orders:
            data = order_data[order]
            for prob, (phase, pauli) in zip(data["probs"], data["terms"]):
                if data["kind"] == "pair":
                    tilde_v_local += raw_weights[order] * prob * compensation_unitary(phase * pauli, theta_pair)
                else:
                    tilde_v_local += raw_weights[order] * prob * (phase * pauli)
        return tilde_v_local

    def tilde_V_taylor():
        tilde_v = np.eye(2**n, dtype=complex)
        for order in s_orders:
            tilde_v += tilde_f_terms[order] * (t**order)
        return tilde_v

    tilde_v_check = tilde_V()
    tilde_v_taylor = tilde_V_taylor()
    print("tilde_V compensate-vs-Taylor:", np.linalg.norm(tilde_v - tilde_v_taylor, 2))
    print("tilde_V cached-vs-recomputed:", np.linalg.norm(tilde_v - tilde_v_check, 2))

    # Sampling start here
    def NCC_sampling(num_trials):
        v_list = []
        for _ in tqdm(range(num_trials), desc="single step trials"):
            order = int(rng.choice(s_orders, p=p_order))
            v_list.append(raw_total * sample_component(order))
        v_average = sum(v_list) / num_trials
        return v_list, v_average

    single_step_error_before = np.linalg.norm(s1 - expm(-1j * (a_mat + b_mat) * t), 2)
    print("single-step error before:", single_step_error_before)

    v_list, v_avg = NCC_sampling(trials)
    single_step_fluctuation = np.linalg.norm(v_avg - tilde_v, 2)
    single_step_sample_error = np.linalg.norm(v_avg - v_exact, 2)
    single_step_expectation_bias = np.linalg.norm(tilde_v - v_exact, 2)

    print("single-step sample error after compensation:", single_step_sample_error)
    print("single-step sample fluctuation:", single_step_fluctuation)
    print("single-step expectation bias:", single_step_expectation_bias)

    def multi_step_NCC_sampling(num_trials):
        evo_list = []
        for _ in tqdm(range(num_trials), desc="multi step trials"):
            evo = np.eye(2**n, dtype=complex)
            for _ in range(r):
                order = int(rng.choice(s_orders, p=p_order))
                evo = (raw_total * sample_component(order)) @ s1 @ evo
            evo_list.append(evo)
        evo_average = sum(evo_list) / num_trials
        return evo_list, evo_average

    total_error_before = np.linalg.norm(np.linalg.matrix_power(s1, r) - u_exact, 2)
    print("total error before:", total_error_before)

    evo_list, evo_avg = multi_step_NCC_sampling(trials)

    total_sample_fluctuation = np.linalg.norm(evo_avg - np.linalg.matrix_power(tilde_v @ s1, r), 2)
    total_sample_error = np.linalg.norm(evo_avg - u_exact, 2)
    total_expectation_bias = np.linalg.norm(np.linalg.matrix_power(tilde_v @ s1, r) - u_exact, 2)

    print("multi-step sample error after compensation:", total_sample_error)
    print("multi-step sample fluctuation:", total_sample_fluctuation)
    print("multi-step expectation bias:", total_expectation_bias)

    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    out = data_dir / f"results_log_{args.sampling}_trials{trials}_s0{s0}.npz"
    np.savez(
        out,
        A=a_mat,
        B=b_mat,
        S=s1,
        V_tilde=tilde_v,
        V_tilde_taylor=tilde_v_taylor,
        V_exact=v_exact,
        V_average=v_avg,
        evolution_average=evo_avg,
        phi_orders=np.array(s_orders, dtype=int),
        phi_matrices=np.stack([phi_terms[s] for s in s_orders]),
        tilde_f_matrices=np.stack([tilde_f_terms[s] for s in s_orders]),
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
        theta=theta_pair,
        eta_sum=eta_pair_sum,
        raw_total=raw_total,
        sampling=args.sampling,
        cond_bch_truncation=cond_bch_truncation,
        cond_finite_s_truncation=cond_finite_s_truncation,
    )
    print("saving results to:", out)


if __name__ == "__main__":
    main()
