import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.linalg import expm
from tqdm import tqdm

from NCC_original import build_periodic_ab, commutator, pauli_basis, pauli_decomposition


@dataclass
class OriginalStaticData:
    n: int
    a_mat: np.ndarray
    b_mat: np.ndarray
    basis: list[np.ndarray]
    c1_coeffs: np.ndarray
    c1_terms: list[np.ndarray]
    c1_l1: float
    c2_coeffs: np.ndarray
    c2_terms: list[np.ndarray]
    c2_l1: float
    identity: np.ndarray
    h_total: np.ndarray


def build_parser():
    parser = argparse.ArgumentParser(description="Sampling-based r_min search for original-NCC.")
    parser.add_argument("--out-dir", type=Path, default=Path("data"), help="output directory")
    parser.add_argument("--tag", type=str, default="", help="optional suffix for output file names")
    parser.add_argument("--N", type=int, default=6, help="number of spins")
    parser.add_argument("--J", type=float, default=1.0, help="interaction strength")
    parser.add_argument("--h", type=float, default=1.0, help="transverse field strength")
    parser.add_argument("--T", type=float, default=1.0, help="total evolution time")
    parser.add_argument("--epsilon", type=float, default=0.01, help="target precision")
    parser.add_argument("--trials", type=int, default=1000, help="Monte Carlo trajectories per r")
    parser.add_argument("--repeats", type=int, default=10, help="number of repeated r_min searches")
    parser.add_argument("--seed", type=int, default=7, help="base RNG seed")
    parser.add_argument("--r-max", type=int, default=512, help="maximal r allowed during search")
    parser.add_argument("--save-every-eval", action="store_true", help="checkpoint after every sampled r evaluation")
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def build_static_data(n: int, j: float, h: float) -> OriginalStaticData:
    a_mat, b_mat = build_periodic_ab(n, j, h)
    basis = pauli_basis(n)
    c1 = commutator(b_mat, a_mat)
    c2 = 1j * (2 * commutator(b_mat, commutator(a_mat, b_mat)) + commutator(a_mat, commutator(a_mat, b_mat)))
    c1_coeffs, c1_terms, c1_l1 = pauli_decomposition(c1, basis, antihermitian=True)
    c2_coeffs, c2_terms, c2_l1 = pauli_decomposition(c2, basis, antihermitian=True)
    dim = 2**n
    return OriginalStaticData(
        n=n,
        a_mat=a_mat,
        b_mat=b_mat,
        basis=basis,
        c1_coeffs=c1_coeffs,
        c1_terms=c1_terms,
        c1_l1=c1_l1,
        c2_coeffs=c2_coeffs,
        c2_terms=c2_terms,
        c2_l1=c2_l1,
        identity=np.eye(dim, dtype=complex),
        h_total=a_mat + b_mat,
    )


def build_step_data(static: OriginalStaticData, t_total: float, r: int):
    t = t_total / r
    eta2 = static.c1_l1 * (t**2) / 2
    eta3 = static.c2_l1 * (t**3) / 6
    eta_sum = eta2 + eta3
    if eta_sum <= 0:
        raise ValueError("eta_sum must be positive")
    p_s = np.array([eta2 / eta_sum, eta3 / eta_sum], dtype=float)
    s1 = expm(-1j * static.b_mat * t) @ expm(-1j * static.a_mat * t)
    u_exact = expm(-1j * static.h_total * t_total)

    tilde_v = np.zeros_like(static.identity)
    for weight, coeffs, terms, l1_norm in (
        (p_s[0], static.c1_coeffs, static.c1_terms, static.c1_l1),
        (p_s[1], static.c2_coeffs, static.c2_terms, static.c2_l1),
    ):
        probs = np.abs(coeffs) / l1_norm
        for prob, coeff, pauli in zip(probs, coeffs, terms):
            sign = coeff / (1j * abs(coeff))
            w_mat = sign * pauli
            tilde_v += weight * prob * (static.identity + 1j * eta_sum * w_mat)

    return {
        "t": t,
        "p_s": p_s,
        "eta_sum": eta_sum,
        "eta2": eta2,
        "eta3": eta3,
        "s1": s1,
        "u_exact": u_exact,
        "tilde_v": tilde_v,
    }


def sample_Pauli_then_compensate_exp(
    rng: np.random.Generator,
    static: OriginalStaticData,
    p_s: np.ndarray,
    eta_sum: float,
    atol: float = 1e-10,
) -> np.ndarray:
    order = int(rng.choice(2, p=p_s))
    if order == 0:
        probs = np.abs(static.c1_coeffs) / static.c1_l1
        idx = int(rng.choice(len(static.c1_terms), p=probs))
        coeff = static.c1_coeffs[idx]
        pauli = static.c1_terms[idx]
    else:
        probs = np.abs(static.c2_coeffs) / static.c2_l1
        idx = int(rng.choice(len(static.c2_terms), p=probs))
        coeff = static.c2_coeffs[idx]
        pauli = static.c2_terms[idx]
    sign = coeff / (1j * abs(coeff))
    w_mat = sign * pauli
    hermitian_err = np.linalg.norm(w_mat - w_mat.conj().T, ord="fro")
    if hermitian_err > atol:
        raise ValueError(f"sampled W is not Hermitian (herm_err={hermitian_err:.3e})")
    return static.identity + 1j * eta_sum * w_mat


def estimate_total_sample_error(static: OriginalStaticData, t_total: float, r: int, trials: int, seed: int):
    step = build_step_data(static, t_total, r)
    rng = np.random.default_rng(seed)
    evo_average = np.zeros_like(static.identity)
    for _ in tqdm(range(trials), desc=f"r={r}", leave=False, disable=not sys.stderr.isatty()):
        evo = static.identity.copy()
        for _ in range(r):
            evo = sample_Pauli_then_compensate_exp(rng, static, step["p_s"], step["eta_sum"]) @ step["s1"] @ evo
        evo_average += evo
    evo_average /= trials

    deterministic = np.linalg.matrix_power(step["tilde_v"] @ step["s1"], r)
    return {
        "sample_error": float(np.linalg.norm(evo_average - step["u_exact"], 2)),
        "sample_fluctuation": float(np.linalg.norm(evo_average - deterministic, 2)),
        "expectation_bias": float(np.linalg.norm(deterministic - step["u_exact"], 2)),
        "t": float(step["t"]),
        "theta": float("nan"),
        "eta_sum": float(step["eta_sum"]),
        "eta2": float(step["eta2"]),
        "eta3": float(step["eta3"]),
    }


def make_eval_seed(base_seed: int, repetition: int, r: int) -> int:
    return base_seed + repetition * 100_003 + r * 1_009


def confidence_interval(values: np.ndarray) -> tuple[float, float, float]:
    mean = float(np.mean(values))
    if len(values) <= 1:
        return mean, mean, mean
    std = float(np.std(values, ddof=1))
    half_width = 1.96 * std / math.sqrt(len(values))
    return mean, mean - half_width, mean + half_width


def resolve_output_dir(base_out_dir: Path, tag: str) -> Path:
    out_dir = base_out_dir / "smoke" if "smoke" in tag.lower() else base_out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def normalized_tag_suffix(tag: str) -> str:
    if not tag:
        return ""
    if "smoke" in tag.lower():
        return "_smoke"
    return f"_{tag}"


def sampling_out_base(args) -> Path:
    suffix = normalized_tag_suffix(args.tag)
    out_dir = resolve_output_dir(args.out_dir, args.tag)
    return out_dir / (f"NCC_original_sampling_r_N{args.N}_T{args.T:g}_eps{args.epsilon:g}_" f"trials{args.trials}_repeats{args.repeats}{suffix}")


def save_sampling_checkpoint(out_base: Path, payload: dict):
    json_path = Path(f"{out_base}.json")
    npz_path = Path(f"{out_base}.npz")
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    np.savez(
        npz_path,
        sampled_r_mins=np.array(payload["sampled_r_mins"], dtype=int),
        expected_r_mins=np.array(payload["expected_r_mins"], dtype=int),
        sample_errors=np.array(payload["sample_errors"], dtype=float),
        expectation_biases=np.array(payload["expectation_biases"], dtype=float),
        sample_fluctuations=np.array(payload["sample_fluctuations"], dtype=float),
        low_bounds=np.array(payload["ci_low_history"], dtype=float),
        high_bounds=np.array(payload["ci_high_history"], dtype=float),
        eval_repetition=np.array([item["repetition"] for item in payload["evaluations"]], dtype=int),
        eval_r=np.array([item["r"] for item in payload["evaluations"]], dtype=int),
        eval_seed=np.array([item["seed"] for item in payload["evaluations"]], dtype=int),
        eval_sample_error=np.array([item["sample_error"] for item in payload["evaluations"]], dtype=float),
        eval_expectation_bias=np.array([item["expectation_bias"] for item in payload["evaluations"]], dtype=float),
        eval_sample_fluctuation=np.array([item["sample_fluctuation"] for item in payload["evaluations"]], dtype=float),
    )


def find_r_min_sampling(
    static: OriginalStaticData,
    t_total: float,
    epsilon: float,
    trials: int,
    repetition: int,
    base_seed: int,
    r_max: int,
    evaluations: list[dict],
    progress_label: str,
    checkpoint_cb=None,
):
    cache: dict[int, dict] = {}

    def evaluate(r: int):
        if r in cache:
            return cache[r]
        seed = make_eval_seed(base_seed, repetition, r)
        result = estimate_total_sample_error(static, t_total, r, trials, seed)
        result["seed"] = seed
        cache[r] = result
        evaluations.append({"repetition": repetition, "r": r, **result})
        print(
            f"{progress_label} eval r={r}: sample_error={result['sample_error']:.6e}, " f"bias={result['expectation_bias']:.6e}, fluct={result['sample_fluctuation']:.6e}"
        )
        if checkpoint_cb is not None:
            checkpoint_cb()
        return result

    def find_min_by(metric_key: str) -> tuple[int, dict]:
        low = 0
        high = 1
        high_eval = evaluate(high)
        while high_eval[metric_key] > epsilon and high < r_max:
            low = high
            high *= 2
            high_eval = evaluate(high)
        if high_eval[metric_key] > epsilon:
            raise RuntimeError(f"failed to reach epsilon={epsilon} by r={r_max} for {metric_key}")

        while low + 1 < high:
            mid = (low + high) // 2
            mid_eval = evaluate(mid)
            if mid_eval[metric_key] <= epsilon:
                high = mid
                high_eval = mid_eval
            else:
                low = mid
        return high, high_eval

    sampled_r_min, sampled_eval = find_min_by("sample_error")
    expected_r_min, _ = find_min_by("expectation_bias")
    return sampled_r_min, sampled_eval, expected_r_min


def main(argv=None):
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_base = sampling_out_base(args)
    static = build_static_data(args.N, args.J, args.h)
    payload = {
        "script": "NCC_original_sampling_r.py",
        "params": {
            "N": args.N,
            "J": args.J,
            "h": args.h,
            "T": args.T,
            "epsilon": args.epsilon,
            "trials": args.trials,
            "repeats": args.repeats,
            "base_seed": args.seed,
            "r_max": args.r_max,
        },
        "sampled_r_mins": [],
        "expected_r_mins": [],
        "sample_errors": [],
        "expectation_biases": [],
        "sample_fluctuations": [],
        "ci_low_history": [],
        "ci_high_history": [],
        "evaluations": [],
    }

    def checkpoint():
        current = np.array(payload["sampled_r_mins"], dtype=float) if payload["sampled_r_mins"] else np.array([], dtype=float)
        if current.size > 0:
            _, low, high = confidence_interval(current)
        else:
            low = high = float("nan")
        payload["ci_low_history"] = payload["ci_low_history"][: len(payload["sampled_r_mins"])]
        payload["ci_high_history"] = payload["ci_high_history"][: len(payload["sampled_r_mins"])]
        save_sampling_checkpoint(out_base, payload)

    for repetition in range(args.repeats):
        label = f"[repeat {repetition + 1}/{args.repeats}]"
        sampled_r_min, metrics, expected_r_min = find_r_min_sampling(
            static=static,
            t_total=args.T,
            epsilon=args.epsilon,
            trials=args.trials,
            repetition=repetition,
            base_seed=args.seed,
            r_max=args.r_max,
            evaluations=payload["evaluations"],
            progress_label=label,
            checkpoint_cb=checkpoint if args.save_every_eval else None,
        )
        payload["sampled_r_mins"].append(int(sampled_r_min))
        payload["expected_r_mins"].append(-1 if expected_r_min is None else int(expected_r_min))
        payload["sample_errors"].append(float(metrics["sample_error"]))
        payload["expectation_biases"].append(float(metrics["expectation_bias"]))
        payload["sample_fluctuations"].append(float(metrics["sample_fluctuation"]))
        _, ci_low, ci_high = confidence_interval(np.array(payload["sampled_r_mins"], dtype=float))
        payload["ci_low_history"].append(ci_low)
        payload["ci_high_history"].append(ci_high)
        save_sampling_checkpoint(out_base, payload)
        print(
            f"{label} sampled_r_min={sampled_r_min}, expected_r_min={expected_r_min}, current mean={np.mean(payload['sampled_r_mins']):.3f}, "
            f"95% CI=[{ci_low:.3f}, {ci_high:.3f}]"
        )

    sampled_r_mins = np.array(payload["sampled_r_mins"], dtype=float)
    mean, ci_low, ci_high = confidence_interval(sampled_r_mins)
    print("finished search")
    print(f"sampled r_min samples: {payload['sampled_r_mins']}")
    print(f"expected r_min samples: {payload['expected_r_mins']}")
    print(f"sampled r_min mean: {mean:.3f}")
    print(f"95% CI for sampled r_min mean: [{ci_low:.3f}, {ci_high:.3f}]")
    print(f"saved checkpoint to: {Path(f'{out_base}.json')}")
    print(f"saved arrays to: {Path(f'{out_base}.npz')}")


if __name__ == "__main__":
    main()
