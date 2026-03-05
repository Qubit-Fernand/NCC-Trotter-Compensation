def gen_q_tuples(j, s, q_min, q_max):
    """生成并迭代所有长度为 j 的整数元组 (q1,...,qj)，满足
    q_min <= qi <= q_max 且 sum(qi) == s。"""
    if j <= 0:
        if s == 0:
            yield ()
        return

    def helper(prefix, remaining_j, remaining_s):
        if remaining_j == 1:
            if q_min <= remaining_s <= q_max:
                yield tuple(prefix + [remaining_s])
            return
        # 给下一个值做剪枝：剩余和必须在可行范围内
        min_next = q_min
        max_next = q_max
        for val in range(min_next, max_next + 1):
            rem = remaining_s - val
            min_rem = q_min * (remaining_j - 1)
            max_rem = q_max * (remaining_j - 1)
            if rem < min_rem or rem > max_rem:
                continue
            yield from helper(prefix + [val], remaining_j - 1, rem)

    yield from helper([], j, s)


gen_q_tuples(2, s, q_min=K + 1, q_max=q_0)


# Build the Ising-model Hamiltonian
class IsingHamiltonian:
    def __init__(self, num_bits, J=0.5, h=1, Heisenberg=False):
        self.num_bits = num_bits
        self.J = J
        self.h = h
        self.Heisenberg = Heisenberg

    def build(self):
        num_bits = self.num_bits
        J = self.J
        h = self.h
        Heisenberg = self.Heisenberg
        H1 = np.zeros((2**num_bits, 2**num_bits), dtype=complex)
        H2 = np.zeros((2**num_bits, 2**num_bits), dtype=complex)
        H3 = np.zeros((2**num_bits, 2**num_bits), dtype=complex)

        # Add nearest-neighbor interaction terms (σz σz)
        for i in range(
            0, num_bits - 1, 2
        ):  # Only nearest neighbors; no periodic boundary
            # Build the σz σz term for the (i, i+1) pair
            # Heisenberg interaction terms can be σx σx, σy σy, or σz σz
            interaction_term = np.kron(
                np.eye(2**i), np.kron(np.kron(Z, Z), np.eye(2 ** (num_bits - i - 2)))
            )
            if Heisenberg:
                interaction_term += np.kron(
                    np.eye(2**i),
                    np.kron(np.kron(X, X), np.eye(2 ** (num_bits - i - 2))),
                )
                interaction_term += np.kron(
                    np.eye(2**i),
                    np.kron(np.kron(Y, Y), np.eye(2 ** (num_bits - i - 2))),
                )
            H3 += -J * interaction_term

        for i in range(1, num_bits - 1, 2):
            # Build the σz σz term for the (i, i+1) pair
            interaction_term = np.kron(
                np.eye(2**i), np.kron(np.kron(Z, Z), np.eye(2 ** (num_bits - i - 2)))
            )
            if Heisenberg:
                interaction_term += np.kron(
                    np.eye(2**i),
                    np.kron(np.kron(X, X), np.eye(2 ** (num_bits - i - 2))),
                )
                interaction_term += np.kron(
                    np.eye(2**i),
                    np.kron(np.kron(Y, Y), np.eye(2 ** (num_bits - i - 2))),
                )
            H2 += -J * interaction_term

        # Add transverse-field terms (σx)
        for i in range(num_bits):
            field_term = np.kron(
                np.eye(2**i), np.kron(X, np.eye(2 ** (num_bits - i - 1)))
            )
            H1 += -h * field_term

        return H1, H2, H3, H1 + H2 + H3
