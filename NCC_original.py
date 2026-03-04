import argparse
from pathlib import Path

import numpy as np
from scipy.linalg import expm
from tqdm import tqdm


# Define Pauli matrices
X = np.array([[0, 1], [1, 0]])  # Pauli-X matrix
Y = np.array([[0, -1j], [1j, 0]])  # Pauli-Y matrix
Z = np.array([[1, 0], [0, -1]])  # Pauli-Z matrix
I = np.eye(2)  # Identity matrix


def commutator(A, B):
    """Compute the commutator [A, B] = AB - BA"""
    return A @ B - B @ A


def _str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"1", "true", "t", "yes", "y"}:
        return True
    if value in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("expected a boolean")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--N", type=int, default=6, help="number of spins")
    parser.add_argument("--J", type=float, default=1.0, help="interaction strength")
    parser.add_argument("--h", type=float, default=1.0, help="transverse field strength")
    parser.add_argument("--T", type=float, default=1.0, help="evolution time")
    parser.add_argument("--r", type=int, default=20, help="Trotter steps")
    # parser.add_argument("--Heisenberg", type=_str2bool, default=True, help="use Heisenberg model (true/false)")
    parser.add_argument("--trials", type=int, default=1000, help="NCC trials")
    return parser.parse_args()


def main():
    args = parse_args()

    # Parameters
    N = args.N
    J = args.J
    h = args.h
    T = args.T
    r = args.r
    Heisenberg = False

    g = 2 * (J + h)  # extensive parameter
    k = 2  # 2-local Hamiltonian

    t = T / r  # step size

    K = 1

    # target accuracy
    epsilon = 0.01

    print("time step:", t)

    if Heisenberg:
        p_Pauli = np.array([J, J, J, h]) / (3 * J + h)
    else:
        p_Pauli = np.array([J, 0, 0, h]) / (J + h)

    def XX_term(index):
        if index < N - 1:
            return J * np.kron(
                np.eye(2**index),
                np.kron(np.kron(X, X), np.eye(2 ** (N - index - 2))),
            )
        if index == N - 1:  # periodic boundary condition
            return np.kron(X, np.kron(np.eye(2 ** (N - 2)), X))
        raise IndexError("out of range")

    def YY_term(index):
        if not Heisenberg:
            return np.zeros((2**N, 2**N), dtype=complex)
        if index < N - 1:
            return J * np.kron(
                np.eye(2**index),
                np.kron(np.kron(Y, Y), np.eye(2 ** (N - index - 2))),
            )
        if index == N - 1:  # periodic boundary condition
            return np.kron(Y, np.kron(np.eye(2 ** (N - 2)), Y))
        raise IndexError("out of range")

    def ZZ_term(index):
        if not Heisenberg:
            return np.zeros((2**N, 2**N), dtype=complex)
        if index < N - 1:
            return J * np.kron(
                np.eye(2**index),
                np.kron(np.kron(Z, Z), np.eye(2 ** (N - index - 2))),
            )
        if index == N - 1:  # periodic boundary condition
            return np.kron(Z, np.kron(np.eye(2 ** (N - 2)), Z))
        raise IndexError("out of range")

    def Z_term(index):
        if index <= N - 1:
            return h * np.kron(
                np.eye(2**index),
                np.kron(Z, np.eye(2 ** (N - index - 1))),
            )
        raise IndexError("out of range")

    def padding_term(index):
        if index >= N - 1:
            return np.eye(2**N)
        raise IndexError("illegal padding!")

    def pauli_l1_norm(matrix, num_qubits):
        """Compute l1 norm of matrix coefficients in n-qubit Pauli basis."""
        single_basis = [I, X, Y, Z]
        basis = [np.array([[1]], dtype=complex)]
        for _ in range(num_qubits):
            next_basis = []
            for left in basis:
                for right in single_basis:
                    next_basis.append(np.kron(left, right))
            basis = next_basis
        scale = 2**num_qubits
        l1_norm = 0.0
        for P in basis:
            coeff = np.trace(P.conj().T @ matrix) / scale
            l1_norm += abs(coeff)
        return float(l1_norm)

    def antihermitian_pauli_sampler(matrix, num_qubits, tol=1e-10):
        """Build sampler for anti-Hermitian matrix C = i * sum_j a_j P_j (a_j real)."""
        single_basis = [I, X, Y, Z]
        basis = [np.array([[1]], dtype=complex)]
        for _ in range(num_qubits):
            next_basis = []
            for left in basis:
                for right in single_basis:
                    next_basis.append(np.kron(left, right))
            basis = next_basis

        scale = 2**num_qubits
        terms = []
        abs_weights = []
        for P in basis:
            coeff = np.trace(P.conj().T @ matrix) / scale
            # For anti-Hermitian matrix in Hermitian Pauli basis: coeff = i * a, a real.
            a = -1j * coeff
            if abs(a) <= tol:
                continue
            if abs(np.imag(a)) > 1e-7:
                raise ValueError(
                    "anti-Hermitian Pauli decomposition has non-negligible imaginary weight"
                )
            a_real = float(np.real(a))
            if abs(a_real) <= tol:
                continue
            terms.append((1.0 if a_real > 0 else -1.0, P))
            abs_weights.append(abs(a_real))

        total_weight = float(sum(abs_weights))
        if total_weight <= 0:
            raise ValueError("empty Pauli sampler: zero decomposition weight")
        probs = np.array(abs_weights, dtype=float) / total_weight
        return terms, probs, total_weight

    # construct product formula
    A = np.zeros((2**N, 2**N), dtype=complex)
    B = np.zeros((2**N, 2**N), dtype=complex)

    for index in range(0, N, 2):
        A += XX_term(index) + YY_term(index) + ZZ_term(index) + Z_term(index)

    for index in range(1, N, 2):
        B += XX_term(index) + YY_term(index) + ZZ_term(index) + Z_term(index)

    # note the order of exp(A) and exp(B)
    S = expm(-1j * B * t) @ expm(-1j * A * t)
    V_exact = expm(-1j * (A + B) * t) @ expm(1j * A * t) @ expm(1j * B * t)

    C1 = commutator(B, A)
    C2 = 1j * (2 * commutator(B, commutator(A, B)) + commutator(A, commutator(A, B)))
    c1_terms, c1_probs, c1_l1 = antihermitian_pauli_sampler(C1, N)
    c2_terms, c2_probs, c2_l1 = antihermitian_pauli_sampler(C2, N)

    eta2 = c1_l1 * (t**2) / 2
    eta3 = c2_l1 * (t**3) / 6
    eta_sum = eta2 + eta3
    p_s = [eta2 / eta_sum, eta3 / eta_sum]
    theta = np.arctan(eta_sum)

    print("C1 l1:", c1_l1)
    print("C2 l1:", c2_l1)
    print("C1/C2 Pauli terms:", len(c1_terms), len(c2_terms))
    print("eta2, eta3:", eta2, eta3)
    print("p_s:", p_s)
    print("theta:", theta)

    S_r = np.eye(2**N, dtype=complex)
    for _ in range(r):
        S_r = S @ S_r

    evolution_exact = expm(-1j * (A + B) * t * r)

    def compensation_unitary(W, theta, atol=1e-10):
        """Build compensation unitary exp(i theta W) for Hermitian Pauli W."""
        hermitian_err = np.linalg.norm(W - W.conj().T, ord="fro")
        if hermitian_err > atol:
            raise ValueError(f"sampled W is not Hermitian (herm_err={hermitian_err:.3e})")
        return expm(1j * theta * W)

    def sample_W_from_commutator(rng, s):
        """Sample Hermitian Pauli W from strict commutator decomposition."""
        if s == 2:
            idx = int(rng.choice(len(c1_terms), p=c1_probs))
            sign, P = c1_terms[idx]
            return sign * P
        if s == 3:
            idx = int(rng.choice(len(c2_terms), p=c2_probs))
            sign, P = c2_terms[idx]
            return sign * P
        raise ValueError(f"unsupported order s={s}")

    # K = 1, sample from s = 2, 3
    def NCC_sampling(trials):
        rng = np.random.default_rng(seed=7)

        # NCC sampling
        s_list = [2, 3]

        V_list = []
        for _ in tqdm(range(trials), desc="single step trails"):
            s = int(rng.choice(s_list, p=p_s))
            W = sample_W_from_commutator(rng, s)
            V_list.append(compensation_unitary(W, theta))

        V_average = sum(V_list) / trials

        return V_list, V_average

    single_step_error_before = np.linalg.norm(S - expm(-1j * (A + B) * t), 2)
    print("single Trotter step error before compensation:\n", single_step_error_before)

    V_list, V_average = NCC_sampling(trials=args.trials)
    single_step_error_after = np.linalg.norm(V_average - V_exact, 2)
    print("single Trotter step error after compensation:\n", single_step_error_after)

    def multi_step_NCC_sampling(trials):
        rng = np.random.default_rng(seed=7)

        # NCC sampling
        s_list = [2, 3]

        evolution_list = []
        for _ in tqdm(range(trials), desc="multi step trials"):
            evolution = np.eye(2**N, dtype=complex)
            for _ in range(r):
                s = int(rng.choice(s_list, p=p_s))
                W = sample_W_from_commutator(rng, s)
                evolution = compensation_unitary(W, theta) @ S @ evolution

            evolution_list.append(evolution)

        evolution_average = sum(evolution_list) / trials

        return evolution_list, evolution_average

    total_error_before = np.linalg.norm(S_r - evolution_exact, 2)
    print("total evolution error before compensation:\n", total_error_before)

    evolution_list, evolution_average = multi_step_NCC_sampling(trials=args.trials)
    total_error_after = np.linalg.norm(evolution_average - evolution_exact, 2)
    print("total evolution error after compensation:\n", total_error_after)

    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    output_path = data_dir / f"results_{args.trials}.npz"
    print("saving results to:", output_path)
    np.savez(
        output_path,
        A=A,
        B=B,
        S=S,
        V_exact=V_exact,
        V_average=V_average,
        V_list=np.array(V_list, dtype=object),
        evolution_average=evolution_average,
        evolution_list=np.array(evolution_list, dtype=object),
        single_step_error_before=single_step_error_before,
        single_step_error_after=single_step_error_after,
        total_error_before=total_error_before,
        total_error_after=total_error_after,
        N=N,
        J=J,
        h=h,
        T=T,
        r=r,
        Heisenberg=Heisenberg,
        trials=args.trials,
    )


if __name__ == "__main__":
    main()
