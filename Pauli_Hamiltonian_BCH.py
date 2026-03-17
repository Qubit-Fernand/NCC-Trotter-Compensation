import itertools
import math
from functools import lru_cache

import numpy as np


X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
I = np.eye(2, dtype=complex)
PAULI_SINGLE_QUBIT = (I, X, Y, Z)
# index label 0,1,2,3 = I,X,Y,Z


def commutator(a, b):
    return a @ b - b @ a


def build_periodic_ab(n, j, h):
    """Build A and B matrices for the periodic Heisenberg Hamiltonian."""

    def two_local_term(index, pauli):
        if index < n - 1:
            return j * np.kron(
                np.eye(2**index, dtype=complex),
                np.kron(np.kron(pauli, pauli), np.eye(2 ** (n - index - 2), dtype=complex)),
            )
        return j * np.kron(pauli, np.kron(np.eye(2 ** (n - 2), dtype=complex), pauli))

    def z_term(index):
        return h * np.kron(
            np.eye(2**index, dtype=complex),
            np.kron(Z, np.eye(2 ** (n - index - 1), dtype=complex)),
        )

    a_mat = np.zeros((2**n, 2**n), dtype=complex)
    b_mat = np.zeros((2**n, 2**n), dtype=complex)
    for idx in range(0, n, 2):
        a_mat += two_local_term(idx, X) + two_local_term(idx, Y) + two_local_term(idx, Z) + z_term(idx)
    for idx in range(1, n, 2):
        b_mat += two_local_term(idx, X) + two_local_term(idx, Y) + two_local_term(idx, Z) + z_term(idx)
    return a_mat, b_mat


def phi_term(A_mat, B_mat, q_max):
    """Compute Phi_q directly from the BCH commutator formula in the PDF."""

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

    x_ops = [1j * B_mat, 1j * A_mat, -1j * (A_mat + B_mat)]  # X1, X2, X3
    dim = A_mat.shape[0]
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
    return phi_terms


def tilde_F_term(Phi_terms, k_order, q0, s0):
    """Return matrices C_s with \\tilde F_{K,s}(x) = C_s x^s."""

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

    dim = next(iter(Phi_terms.values())).shape[0]
    tilde_F_terms = {}
    q_min = k_order + 1
    for s in range(q_min, s0 + 1):
        total = np.zeros((dim, dim), dtype=complex)
        max_parts = s // q_min
        for j in range(1, max_parts + 1):
            for q_tuple in iter_compositions(s, j, q_min, q0):
                product = np.eye(dim, dtype=complex)
                for q in q_tuple:
                    product = product @ Phi_terms[q]
                total += product / math.factorial(j)
        tilde_F_terms[s] = total
    return tilde_F_terms


def pauli_matrix_from_label(label):
    """Materialize a dense Pauli matrix from a compact integer label."""
    matrix = np.array([[1]], dtype=complex)
    for entry in label:
        matrix = np.kron(matrix, PAULI_SINGLE_QUBIT[entry])
    return matrix


@lru_cache(maxsize=512)
def cached_pauli_matrix_from_label(label):
    """Materialize and cache a dense Pauli matrix from a compact integer label."""
    return pauli_matrix_from_label(label)


def pauli_decomposition_stream(mat, antihermitian=False, tol=1e-10):
    """Decompose mat in the Pauli basis without materializing the full basis list."""

    def label_iter(num_qubits):
        if num_qubits == 0:
            yield ()
            return
        for prefix in label_iter(num_qubits - 1):
            for pauli_idx in range(4):
                yield prefix + (pauli_idx,)

    n = int(round(math.log2(mat.shape[0])))
    scale = 2**n
    coeffs = []
    labels = []
    for label in label_iter(n):
        p = pauli_matrix_from_label(label)
        coeff = np.trace(p.conj().T @ mat) / scale
        if antihermitian:
            if abs(coeff) <= tol:
                continue
            if abs(np.real(coeff)) > 1e-7:
                raise ValueError("anti-Hermitian Pauli decomposition has non-negligible real Pauli coefficient")
        else:
            if abs(coeff) <= tol:
                continue
        coeffs.append(coeff)
        labels.append(label)
    coeffs = np.array(coeffs, dtype=complex)
    l1_norm = float(np.sum(np.abs(coeffs)))
    if l1_norm <= 0:
        kind = "anti-Hermitian Pauli" if antihermitian else "Pauli"
        raise ValueError(f"empty {kind} decomposition")
    return coeffs, labels, l1_norm
