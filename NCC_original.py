"""
We first compute the commutator and then calculate its Pauli 1-norm.
We use the commutator results' Pauli expansion to pair into exp unlike the pseudocode in prx paper, which use the summands in the commutator to pair.
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


def build_periodic_ab(num_qubits, coupling_j, field_h):
    """Build A and B matrices for the periodic Heisenberg Hamiltonian."""

    def two_local_term(index, pauli):
        if index < num_qubits - 1:
            return coupling_j * np.kron(
                np.eye(2**index, dtype=complex),
                np.kron(np.kron(pauli, pauli), np.eye(2 ** (num_qubits - index - 2), dtype=complex)),
            )
        return coupling_j * np.kron(pauli, np.kron(np.eye(2 ** (num_qubits - 2), dtype=complex), pauli))

    def z_term(index):
        return field_h * np.kron(
            np.eye(2**index, dtype=complex),
            np.kron(Z, np.eye(2 ** (num_qubits - index - 1), dtype=complex)),
        )

    a_mat = np.zeros((2**num_qubits, 2**num_qubits), dtype=complex)
    b_mat = np.zeros((2**num_qubits, 2**num_qubits), dtype=complex)
    for index in range(0, num_qubits, 2):
        a_mat += two_local_term(index, X) + two_local_term(index, Y) + two_local_term(index, Z) + z_term(index)
    for index in range(1, num_qubits, 2):
        b_mat += two_local_term(index, X) + two_local_term(index, Y) + two_local_term(index, Z) + z_term(index)
    return a_mat, b_mat


def pauli_decomposition(matrix, basis, antihermitian=False, tol=1e-10):
    """Decompose a matrix in the Pauli basis."""
    num_qubits = int(round(np.log2(matrix.shape[0])))
    scale = 2**num_qubits
    coeffs = []
    terms = []
    for P in basis:
        coeff = np.trace(P.conj().T @ matrix) / scale
        if antihermitian:
            if abs(coeff) <= tol:
                continue
            if abs(np.real(coeff)) > 1e-7:
                raise ValueError("anti-Hermitian Pauli decomposition has non-negligible real Pauli coefficient")
        else:
            if abs(coeff) <= tol:
                continue
        coeffs.append(coeff)
        terms.append(P)

    coeffs = np.array(coeffs, dtype=complex)
    l1_norm = float(np.sum(np.abs(coeffs)))
    if l1_norm <= 0:
        kind = "anti-Hermitian Pauli" if antihermitian else "Pauli"
        raise ValueError(f"empty {kind} decomposition: zero decomposition weight")
    return coeffs, terms, l1_norm


def main():
    args = parse_args()

    # Parameters
    N = args.N
    J = args.J
    h = args.h
    T = args.T
    r = args.r

    g = 2 * (J + h)  # extensive parameter
    k = 2  # 2-local Hamiltonian

    t = T / r  # step size

    K = 1

    # target accuracy
    epsilon = 0.01

    print("time step:", t)

    A, B = build_periodic_ab(N, J, h)
    basis = pauli_basis(N)

    # note the order of exp(A) and exp(B)
    S = expm(-1j * B * t) @ expm(-1j * A * t)
    V_exact = expm(-1j * (A + B) * t) @ expm(1j * A * t) @ expm(1j * B * t)

    C1 = commutator(B, A)
    C2 = 1j * (2 * commutator(B, commutator(A, B)) + commutator(A, commutator(A, B)))
    F_terms = {}
    c1_coeffs, c1_terms, c1_l1 = pauli_decomposition(C1, basis, antihermitian=True)
    c2_coeffs, c2_terms, c2_l1 = pauli_decomposition(C2, basis, antihermitian=True)
    F_terms[2] = {"coeffs": c1_coeffs, "terms": c1_terms, "l1_norm": c1_l1}
    F_terms[3] = {"coeffs": c2_coeffs, "terms": c2_terms, "l1_norm": c2_l1}

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

    def sample_Pauli_then_compensate_exp(rng, s, atol=1e-10):
        """Sample a Hermitian Pauli W from order-s commutator data"""
        F_term = F_terms.get(s)
        if F_term is None:
            raise ValueError(f"unsupported order s={s}")
        probs = np.abs(F_term["coeffs"]) / F_term["l1_norm"]
        idx = int(rng.choice(len(F_term["terms"]), p=probs))
        # For antiHermtian F_2 and F_3, the coeffs are purely imaginary, so W is Hermitian.
        coeff = F_term["coeffs"][idx]
        pauli = F_term["terms"][idx]

        # With prob, sample I + eta_sum * (\pm 1j) * pauli, note the sign
        sign = coeff / (1j * abs(coeff))
        w_mat = sign * pauli
        hermitian_err = np.linalg.norm(w_mat - w_mat.conj().T, ord="fro")
        if hermitian_err > atol:
            raise ValueError(f"sampled W is not Hermitian (herm_err={hermitian_err:.3e})")

        # aim to apply exp(i theta W) * \sqrt{1+eta_sum^2}.
        # numerically we equivalently apply the term before pairing
        return I + 1j * eta_sum * w_mat

    def tilde_V():
        """Expectation of single-step compensation unitary"""
        tilde_v = np.zeros((2**N, 2**N), dtype=complex)
        for s, p_order in zip([2, 3], p_s):
            F_term = F_terms[s]
            probs = np.abs(F_term["coeffs"]) / F_term["l1_norm"]
            for p_pauli, coeff, pauli in zip(probs, F_term["coeffs"], F_term["terms"]):
                sign = coeff / (1j * abs(coeff))
                w_mat = sign * pauli
                tilde_v += p_order * p_pauli * (I + 1j * eta_sum * w_mat)
        return tilde_v

    # K = 1, sample from s = 2, 3
    def NCC_sampling(trials):
        rng = np.random.default_rng(seed=7)

        # NCC sampling
        s_list = [2, 3]

        V_list = []
        for _ in tqdm(range(trials), desc="single step trails"):
            s = int(rng.choice(s_list, p=p_s))
            V_list.append(sample_Pauli_then_compensate_exp(rng, s))

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
                evolution = sample_Pauli_then_compensate_exp(rng, s) @ S @ evolution

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
        trials=args.trials,
    )


if __name__ == "__main__":
    main()
