import argparse
from pathlib import Path

import numpy as np
from scipy.linalg import expm


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
    parser.add_argument(
        "--h", type=float, default=1.0, help="transverse field strength"
    )
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

    U = np.eye(2**N, dtype=complex)
    for _ in range(r):
        U = S @ U
    evolution_exact = expm(-1j * (A + B) * t * r)

    # K = 1, sample from s = 2, 3
    def NCC_sampling(trials):
        rng = np.random.default_rng(seed=7)

        # NCC sampling
        s_list = [2, 3]
        p_s = [1 / (1 + 24 * t), 24 * t / (1 + 24 * t)]

        V_list = []
        for _ in range(trials):
            s = int(rng.choice(s_list, p=p_s))

            j = rng.choice(np.arange(1, N, 2))
            W = rng.choice([XX_term(j), YY_term(j), ZZ_term(j), Z_term(j)], p=p_Pauli)
            j1 = rng.choice([j - 1, j + 1]) % N
            P1 = rng.choice(
                [XX_term(j1), YY_term(j1), ZZ_term(j1), Z_term(j1)], p=p_Pauli
            )
            b1 = rng.choice([0, 1])
            if b1 == 0:
                W = P1 @ W
            else:
                W = -W @ P1

            if s == 3:
                if rng.random() <= 1 / 3:
                    j2 = rng.choice([j - 2, j, j + 2]) % N
                else:
                    j2 = rng.choice([j - 3, j - 1, j + 1]) % N
                P2 = rng.choice(
                    [XX_term(j2), YY_term(j2), ZZ_term(j2), Z_term(j2)], p=p_Pauli
                )
                b2 = rng.choice([0, 1])

                if b2 == 0:
                    W = P2 @ W
                else:
                    W = -W @ P2

            theta = np.arctan(16 * N * pow(t, 2) * (1 + 24 * t))
            V_list.append(expm(1j * theta * W))

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
        p_s = [1 / (1 + 24 * t), 24 * t / (1 + 24 * t)]

        evolution_list = []
        for _ in range(trials):
            evolution = np.eye(2**N, dtype=complex)
            for _ in range(r):
                s = int(rng.choice(s_list, p=p_s))

                j = rng.choice(np.arange(1, N, 2))
                W = rng.choice(
                    [XX_term(j), YY_term(j), ZZ_term(j), Z_term(j)], p=p_Pauli
                )
                j1 = rng.choice([j - 1, j + 1]) % N
                P1 = rng.choice(
                    [XX_term(j1), YY_term(j1), ZZ_term(j1), Z_term(j1)], p=p_Pauli
                )
                b1 = rng.choice([0, 1])
                if b1 == 0:
                    W = P1 @ W
                else:
                    W = -W @ P1

                if s == 3:
                    if rng.random() <= 1 / 3:
                        j2 = rng.choice([j - 2, j, j + 2]) % N
                    else:
                        j2 = rng.choice([j - 3, j - 1, j + 1]) % N
                    P2 = rng.choice(
                        [XX_term(j2), YY_term(j2), ZZ_term(j2), Z_term(j2)],
                        p=p_Pauli,
                    )
                    b2 = rng.choice([0, 1])

                    if b2 == 0:
                        W = P2 @ W
                    else:
                        W = -W @ P2

                theta = np.arctan(16 * N * pow(t, 2) * (1 + 24 * t))
                evolution = expm(1j * theta * W) @ S @ evolution

            evolution_list.append(evolution)

        evolution_average = sum(evolution_list) / trials

        return evolution_list, evolution_average

    total_error_before = np.linalg.norm(U - evolution_exact, 2)
    print("total evolution error before compensation:\n", total_error_before)

    evolution_list, evolution_average = multi_step_NCC_sampling(trials=args.trials)
    total_error_after = np.linalg.norm(evolution_average - evolution_exact, 2)
    print("total evolution error after compensation:\n", total_error_after)

    data_dir = Path("data")
    data_dir.mkdir(parents=True, exist_ok=True)
    output_path = data_dir / "results.npz"
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
