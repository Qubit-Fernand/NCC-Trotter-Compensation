"""
We first compute the commutator and then calculate its Pauli 1-norm.
We use the commutator results to pair into exp unlike the pseudocode in prx paper, which use the summands in the commutator to pair.
"""

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


def pauli_basis(num_qubits):
    single_basis = [I, X, Y, Z]
    basis = [np.array([[1]], dtype=complex)]
    for _ in range(num_qubits):
        next_basis = []
        for left in basis:
            for right in single_basis:
                next_basis.append(np.kron(left, right))
        basis = next_basis
    return basis


def build_periodic_ab(num_qubits, coupling_j, field_h, heisenberg=False):
    def xx_term(index):
        if index < num_qubits - 1:
            return coupling_j * np.kron(
                np.eye(2**index),
                np.kron(np.kron(X, X), np.eye(2 ** (num_qubits - index - 2))),
            )
        if index == num_qubits - 1:
            return np.kron(X, np.kron(np.eye(2 ** (num_qubits - 2)), X))
        raise IndexError("out of range")

    def yy_term(index):
        if not heisenberg:
            return np.zeros((2**num_qubits, 2**num_qubits), dtype=complex)
        if index < num_qubits - 1:
            return coupling_j * np.kron(
                np.eye(2**index),
                np.kron(np.kron(Y, Y), np.eye(2 ** (num_qubits - index - 2))),
            )
        if index == num_qubits - 1:
            return np.kron(Y, np.kron(np.eye(2 ** (num_qubits - 2)), Y))
        raise IndexError("out of range")

    def zz_term(index):
        if not heisenberg:
            return np.zeros((2**num_qubits, 2**num_qubits), dtype=complex)
        if index < num_qubits - 1:
            return coupling_j * np.kron(
                np.eye(2**index),
                np.kron(np.kron(Z, Z), np.eye(2 ** (num_qubits - index - 2))),
            )
        if index == num_qubits - 1:
            return np.kron(Z, np.kron(np.eye(2 ** (num_qubits - 2)), Z))
        raise IndexError("out of range")

    def z_term(index):
        if index <= num_qubits - 1:
            return field_h * np.kron(
                np.eye(2**index),
                np.kron(Z, np.eye(2 ** (num_qubits - index - 1))),
            )
        raise IndexError("out of range")

    a_mat = np.zeros((2**num_qubits, 2**num_qubits), dtype=complex)
    b_mat = np.zeros((2**num_qubits, 2**num_qubits), dtype=complex)
    for index in range(0, num_qubits, 2):
        a_mat += xx_term(index) + yy_term(index) + zz_term(index) + z_term(index)
    for index in range(1, num_qubits, 2):
        b_mat += xx_term(index) + yy_term(index) + zz_term(index) + z_term(index)
    return a_mat, b_mat


def pauli_decomposition(matrix, basis, antihermitian=False, tol=1e-10):
    """Decompose a matrix in the Pauli basis."""
    num_qubits = int(round(np.log2(matrix.shape[0])))
    scale = 2**num_qubits
    terms = []
    abs_weights = []
    for P in basis:
        coeff = np.trace(P.conj().T @ matrix) / scale
        if antihermitian:
            a = -1j * coeff
            if abs(a) <= tol:
                continue
            if abs(np.imag(a)) > 1e-7:
                raise ValueError("anti-Hermitian Pauli decomposition has non-negligible imaginary weight")
            a_real = float(np.real(a))
            if abs(a_real) <= tol:
                continue
            terms.append((1.0 if a_real > 0 else -1.0, P))
            abs_weights.append(abs(a_real))
        else:
            if abs(coeff) <= tol:
                continue
            coeff_abs = abs(coeff)
            terms.append((coeff / coeff_abs, P))
            abs_weights.append(coeff_abs)

    total_weight = float(sum(abs_weights))
    if total_weight <= 0:
        kind = "anti-Hermitian Pauli" if antihermitian else "Pauli"
        raise ValueError(f"empty {kind} decomposition: zero decomposition weight")
    probs = np.array(abs_weights, dtype=float) / total_weight
    return terms, probs, total_weight


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

    A, B = build_periodic_ab(N, J, h, heisenberg=Heisenberg)
    basis = pauli_basis(N)

    # note the order of exp(A) and exp(B)
    S = expm(-1j * B * t) @ expm(-1j * A * t)
    V_exact = expm(-1j * (A + B) * t) @ expm(1j * A * t) @ expm(1j * B * t)

    C1 = commutator(B, A)
    C2 = 1j * (2 * commutator(B, commutator(A, B)) + commutator(A, commutator(A, B)))
    order_data = {}
    c1_terms, c1_probs, c1_l1 = pauli_decomposition(C1, basis, antihermitian=True)
    c2_terms, c2_probs, c2_l1 = pauli_decomposition(C2, basis, antihermitian=True)
    order_data[2] = {"terms": c1_terms, "probs": c1_probs, "l1_norm": c1_l1}
    order_data[3] = {"terms": c2_terms, "probs": c2_probs, "l1_norm": c2_l1}

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

    def compensation_unitary(w_mat, angle, atol=1e-10):
        """Build compensation unitary exp(i angle W) for Hermitian Pauli W."""
        hermitian_err = np.linalg.norm(w_mat - w_mat.conj().T, ord="fro")
        if hermitian_err > atol:
            raise ValueError(f"sampled W is not Hermitian (herm_err={hermitian_err:.3e})")
        return expm(1j * angle * w_mat)

    def sample_W_from_commutator(rng, s):
        """Sample Hermitian Pauli W from strict commutator decomposition."""
        data = order_data.get(s)
        if data is None:
            raise ValueError(f"unsupported order s={s}")
        idx = int(rng.choice(len(data["terms"]), p=data["probs"]))
        sign, pauli = data["terms"][idx]
        return sign * pauli

    def sample_component(rng, s):
        return compensation_unitary(sample_W_from_commutator(rng, s), theta)

    def tilde_V():
        tilde_v = np.zeros((2**N, 2**N), dtype=complex)
        for s, p in zip([2, 3], p_s):
            data = order_data[s]
            for prob, (sign, pauli) in zip(data["probs"], data["terms"]):
                tilde_v += p * prob * compensation_unitary(sign * pauli, theta)
        return tilde_v

    # K = 1, sample from s = 2, 3
    def NCC_sampling(trials):
        rng = np.random.default_rng(seed=7)

        # NCC sampling
        s_list = [2, 3]

        V_list = []
        for _ in tqdm(range(trials), desc="single step trails"):
            s = int(rng.choice(s_list, p=p_s))
            V_list.append(sample_component(rng, s))

        V_average = sum(V_list) / trials

        return V_list, V_average

    single_step_error_before = np.linalg.norm(S - expm(-1j * (A + B) * t), 2)
    print("single Trotter step error before compensation:\n", single_step_error_before)

    tilde_v = tilde_V()
    V_list, V_average = NCC_sampling(trials=args.trials)
    single_step_error_after = np.linalg.norm(V_average - V_exact, 2)
    single_step_fluctuation = np.linalg.norm(V_average - tilde_v, 2)
    single_step_bias = np.linalg.norm(tilde_v - V_exact, 2)

    print("single step error after compensation:\n", single_step_error_after)
    print("single step fluctuation:\n", single_step_fluctuation)
    print("single step expectation bias:\n", single_step_bias)

    def multi_step_NCC_sampling(trials):
        rng = np.random.default_rng(seed=7)

        # NCC sampling
        s_list = [2, 3]

        evolution_list = []
        for _ in tqdm(range(trials), desc="multi step trials"):
            evolution = np.eye(2**N, dtype=complex)
            for _ in range(r):
                s = int(rng.choice(s_list, p=p_s))
                evolution = sample_component(rng, s) @ S @ evolution

            evolution_list.append(evolution)

        evolution_average = sum(evolution_list) / trials

        return evolution_list, evolution_average

    total_error_before = np.linalg.norm(S_r - evolution_exact, 2)
    print("total evolution error before compensation:\n", total_error_before)

    evolution_list, evolution_average = multi_step_NCC_sampling(trials=args.trials)
    total_error_after = np.linalg.norm(evolution_average - evolution_exact, 2)
    total_fluctuation = np.linalg.norm(evolution_average - np.linalg.matrix_power(tilde_v @ S, r), 2)
    total_bias = np.linalg.norm(np.linalg.matrix_power(tilde_v @ S, r) - evolution_exact, 2)

    print("total evolution error after compensation:\n", total_error_after)
    print("total evolution fluctuation:\n", total_fluctuation)
    print("total evolution expectation bias:\n", total_bias)

    data_dir = Path("data_local_unitary_list")
    data_dir.mkdir(parents=True, exist_ok=True)
    output_path = data_dir / f"NCC_original_trials{args.trials}.npz"
    print("saving results to:", output_path)
    np.savez(
        output_path,
        A=A,
        B=B,
        S=S,
        V_tilde=tilde_v,
        V_exact=V_exact,
        V_average=V_average,
        V_list=np.array(V_list, dtype=object),
        evolution_average=evolution_average,
        evolution_list=np.array(evolution_list, dtype=object),
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
        Heisenberg=Heisenberg,
        trials=args.trials,
    )


if __name__ == "__main__":
    main()
