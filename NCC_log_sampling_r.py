import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

from NCC_log import build_static_data, build_tilde_v, build_deterministic_bias


def build_parser():
    parser = argparse.ArgumentParser(description="Sampling-based r_min search for log-NCC.")
    parser.add_argument("--out-dir", type=Path, default=Path("data"), help="output directory")
    parser.add_argument("--tag", type=str, default="", help="optional suffix for output file names")
    parser.add_argument("--N", type=int, default=6, help="number of spins")
    parser.add_argument("--J", type=float, default=1.0, help="interaction strength")
    parser.add_argument("--h", type=float, default=1.0, help="field strength")
    parser.add_argument("--T", type=float, default=1.0, help="total evolution time")
    parser.add_argument("--epsilon", type=float, default=0.01, help="target precision")
    parser.add_argument("--trials", type=int, default=1000, help="Monte Carlo trajectories per r")
    parser.add_argument("--repeats", type=int, default=10, help="number of repeated r_min searches")
    parser.add_argument("--seed", type=int, default=7, help="base RNG seed")
    parser.add_argument("--r-max", type=int, default=512, help="maximal r allowed during search")
    parser.add_argument("--s0", type=int, default=0, help="override log truncation order")
    parser.add_argument("--save-every-eval", action="store_true", help="checkpoint after every sampled r evaluation")
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def make_eval_seed(base_seed: int, repetition: int, r: int) -> int:
    return base_seed + repetition * 100_003 + r * 1_009


def confidence_interval(values: np.ndarray) -> tuple[float, float, float]:
    mean = float(np.mean(values))
    if len(values) <= 1:
        return mean, mean, mean
    std = float(np.std(values, ddof=1))
    half_width = 1.96 * std / math.sqrt(len(values))
    return mean, mean - half_width, mean + half_width


def effective_log_s0(epsilon: float, requested_s0: int) -> int:
    if requested_s0 > 0:
        return max(3, int(requested_s0))
    return max(3, int(np.ceil(np.log(4 / epsilon))))


def effective_log_q0(n: int, epsilon: float) -> int:
    return int(np.ceil(np.log(4 * n / epsilon)))


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
    actual_s0 = effective_log_s0(args.epsilon, args.s0)
    default_s0 = effective_log_s0(args.epsilon, 0)
    s0_suffix = f"_s0{actual_s0}" if actual_s0 != default_s0 else ""
    out_dir = resolve_output_dir(args.out_dir, args.tag)
    return out_dir / (
        f"NCC_log_sampling_r_N{args.N}_T{args.T:g}_eps{args.epsilon:g}_"
        f"trials{args.trials}_repeats{args.repeats}{s0_suffix}{suffix}"
    )


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

def sample_Pauli_then_compensate_exp(
    rng: np.random.Generator,
    identity: np.ndarray,
    f_terms: dict,
    order: int,
    eta_pair_sum: float,
    pair_scale: float,
    atol: float = 1e-10,
):
    data = f_terms[order]
    probs = np.abs(data["coeffs"]) / data["l1_norm"]
    idx = int(rng.choice(len(data["terms"]), p=probs))
    coeff = data["coeffs"][idx]
    pauli = data["terms"][idx]
    if data["kind"] == "pair":
        phase = coeff / (1j * abs(coeff))
        w_mat = phase * pauli
        hermitian_err = np.linalg.norm(w_mat - w_mat.conj().T, ord="fro")
        if hermitian_err > atol:
            raise ValueError(f"sampled W is not Hermitian (herm_err={hermitian_err:.3e})")
        return (identity + 1j * eta_pair_sum * w_mat) / pair_scale
    return coeff / abs(coeff) * pauli


def estimate_total_sample_error(
    n: int,
    t_total: float,
    r: int,
    epsilon: float,
    trials: int,
    seed: int,
    j: float,
    h: float,
    s0: int,
):
    kappa = 1
    static = build_static_data(n, epsilon, j=j, h=h, kappa=kappa, s0=s0 or None)
    step_data = build_tilde_v(static, t_total, r)
    s_orders = static["s_orders"]
    f_terms = static["F_terms"]
    identity = static["identity"]
    raw_weights = step_data["raw_weights"]
    raw_total = float(sum(raw_weights.values()))
    p_order = np.array([raw_weights[s] / raw_total for s in s_orders], dtype=float)

    rng = np.random.default_rng(seed)
    evo_average = np.zeros_like(identity)
    for _ in tqdm(range(trials), desc=f"r={r}", leave=False, disable=not sys.stderr.isatty()):
        evo = identity.copy()
        for _ in range(r):
            order = int(rng.choice(s_orders, p=p_order))
            evo = (
                raw_total
                * sample_Pauli_then_compensate_exp(
                    rng,
                    identity,
                    f_terms,
                    order,
                    step_data["eta_pair_sum"],
                    step_data["pair_scale"],
                )
            ) @ step_data["s1"] @ evo
        evo_average += evo
    evo_average /= trials

    deterministic = np.linalg.matrix_power(step_data["tilde_v"] @ step_data["s1"], r)
    return {
        "sample_error": float(np.linalg.norm(evo_average - step_data["u_exact"], 2)),
        "sample_fluctuation": float(np.linalg.norm(evo_average - deterministic, 2)),
        "expectation_bias": float(np.linalg.norm(deterministic - step_data["u_exact"], 2)),
        "theta_pair": float("nan"),
        "eta_pair_sum": float(step_data["eta_pair_sum"]),
        "pair_scale": float(step_data["pair_scale"]),
        "raw_total": raw_total,
    }


def find_min_segments(n, t_total, epsilon, j=1.0, h=1.0, r_max=512, kappa=1, s0=None):
    """Binary search the smallest r with deterministic log-NCC error <= epsilon."""
    low = 1
    high = 1
    err_high = build_deterministic_bias(n, t_total, high, epsilon, j=j, h=h, kappa=kappa, s0=s0)
    while err_high > epsilon and high < r_max:
        low = high
        high *= 2
        err_high = build_deterministic_bias(
            n,
            t_total,
            high,
            epsilon,
            j=j,
            h=h,
            kappa=kappa,
            s0=s0,
        )
    if err_high > epsilon:
        raise RuntimeError(f"failed to reach epsilon={epsilon} by r={r_max}")

    while low + 1 < high:
        mid = (low + high) // 2
        err_mid = build_deterministic_bias(n, t_total, mid, epsilon, j=j, h=h, kappa=kappa, s0=s0)
        if err_mid <= epsilon:
            high = mid
            err_high = err_mid
        else:
            low = mid
    return high, err_high


def find_r_min_sampling(
    n: int,
    t_total: float,
    epsilon: float,
    trials: int,
    repetition: int,
    base_seed: int,
    r_max: int,
    j: float,
    h: float,
    s0: int,
    evaluations: list[dict],
    progress_label: str,
    checkpoint_cb=None,
):
    cache: dict[int, dict] = {}

    def evaluate(r: int):
        if r in cache:
            return cache[r]
        seed = make_eval_seed(base_seed, repetition, r)
        result = estimate_total_sample_error(
            n=n,
            t_total=t_total,
            r=r,
            epsilon=epsilon,
            trials=trials,
            seed=seed,
            j=j,
            h=h,
            s0=s0,
        )
        result["seed"] = seed
        cache[r] = result
        evaluations.append(
            {
                "repetition": repetition,
                "r": r,
                **result,
            }
        )
        print(
            f"{progress_label} eval r={r}: sample_error={result['sample_error']:.6e}, "
            f"bias={result['expectation_bias']:.6e}, fluct={result['sample_fluctuation']:.6e}"
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
    actual_s0 = effective_log_s0(args.epsilon, args.s0)
    q_0 = effective_log_q0(args.N, args.epsilon)
    payload = {
        "script": "NCC_log_sampling_r.py",
        "params": {
            "N": args.N,
            "J": args.J,
            "h": args.h,
            "Heisenberg": True,
            "T": args.T,
            "epsilon": args.epsilon,
            "trials": args.trials,
            "repeats": args.repeats,
            "base_seed": args.seed,
            "r_max": args.r_max,
            "sampling": "weighted",
            "s0": actual_s0,
            "q_0": q_0,
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

    build_static_data(args.N, args.epsilon, j=args.J, h=args.h, kappa=1, s0=actual_s0)

    def checkpoint():
        save_sampling_checkpoint(out_base, payload)

    for repetition in range(args.repeats):
        label = f"[repeat {repetition + 1}/{args.repeats}]"
        sampled_r_min, metrics, expected_r_min = find_r_min_sampling(
            n=args.N,
            t_total=args.T,
            epsilon=args.epsilon,
            trials=args.trials,
            repetition=repetition,
            base_seed=args.seed,
            r_max=args.r_max,
            j=args.J,
            h=args.h,
            s0=actual_s0,
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
