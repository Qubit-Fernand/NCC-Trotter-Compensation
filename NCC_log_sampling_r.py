import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

from NCC_log import build_log_static_data, build_log_tilde_v


def parse_args():
    parser = argparse.ArgumentParser(description="Sampling-based r_min search for NCC_log.")
    parser.add_argument("--N", type=int, default=6, help="number of spins")
    parser.add_argument("--J", type=float, default=1.0, help="interaction strength")
    parser.add_argument("--h", type=float, default=1.0, help="field strength")
    parser.add_argument("--T", type=float, default=1.0, help="total evolution time")
    parser.add_argument("--epsilon", type=float, default=0.01, help="target precision")
    parser.add_argument("--trials", type=int, default=1000, help="Monte Carlo trajectories per r")
    parser.add_argument("--repeats", type=int, default=10, help="number of repeated r_min searches")
    parser.add_argument("--seed", type=int, default=7, help="base RNG seed")
    parser.add_argument("--r-max", type=int, default=512, help="maximal r allowed during search")
    parser.add_argument("--sampling", choices=["uniform", "weighted"], default="weighted")
    parser.add_argument("--s0", type=int, default=0, help="override log truncation order")
    parser.add_argument("--save-every-eval", action="store_true", help="checkpoint after every sampled r evaluation")
    parser.add_argument("--tag", type=str, default="", help="optional suffix for output file names")
    return parser.parse_args()


def pauli_rotation(pauli: np.ndarray, phase: complex, angle: float, identity: np.ndarray) -> np.ndarray:
    signed_pauli = phase * pauli
    return np.cos(angle) * identity + 1j * np.sin(angle) * signed_pauli


def make_eval_seed(base_seed: int, repetition: int, r: int) -> int:
    return base_seed + repetition * 100_003 + r * 1_009


def confidence_interval(values: np.ndarray) -> tuple[float, float, float]:
    mean = float(np.mean(values))
    if len(values) <= 1:
        return mean, mean, mean
    std = float(np.std(values, ddof=1))
    half_width = 1.96 * std / math.sqrt(len(values))
    return mean, mean - half_width, mean + half_width


def save_checkpoint(out_base: Path, payload: dict):
    json_path = Path(f"{out_base}.json")
    npz_path = Path(f"{out_base}.npz")
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    np.savez(
        npz_path,
        r_mins=np.array(payload["r_mins"], dtype=int),
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


def sample_component(
    rng: np.random.Generator,
    identity: np.ndarray,
    order_data: dict,
    order: int,
    theta_pair: float,
):
    data = order_data[order]
    idx = int(rng.choice(len(data["terms"]), p=data["probs"]))
    phase, pauli = data["terms"][idx]
    if data["kind"] == "pair":
        return pauli_rotation(pauli, phase, theta_pair, identity)
    return phase * pauli


def estimate_total_sample_error(
    n: int,
    t_total: float,
    r: int,
    epsilon: float,
    trials: int,
    seed: int,
    j: float,
    h: float,
    sampling: str,
    s0: int,
):
    kappa = 1
    static = build_log_static_data(n, epsilon, j=j, h=h, sampling=sampling, kappa=kappa, s0=s0 or None)
    step_data = build_log_tilde_v(n, t_total, r, epsilon, j=j, h=h, sampling=sampling, kappa=kappa, s0=s0 or None)
    s_orders = static["s_orders"]
    order_data = static["order_data"]
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
            evo = (raw_total * sample_component(rng, identity, order_data, order, step_data["theta_pair"])) @ step_data["s1"] @ evo
        evo_average += evo
    evo_average /= trials

    deterministic = np.linalg.matrix_power(step_data["tilde_v"] @ step_data["s1"], r)
    return {
        "sample_error": float(np.linalg.norm(evo_average - step_data["u_exact"], 2)),
        "sample_fluctuation": float(np.linalg.norm(evo_average - deterministic, 2)),
        "expectation_bias": float(np.linalg.norm(deterministic - step_data["u_exact"], 2)),
        "theta_pair": float(step_data["theta_pair"]),
        "eta_pair_sum": float(step_data["eta_pair_sum"]),
        "raw_total": raw_total,
    }


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
    sampling: str,
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
            sampling=sampling,
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

    low = 0
    high = 1
    high_eval = evaluate(high)
    while high_eval["sample_error"] > epsilon and high < r_max:
        low = high
        high *= 2
        high_eval = evaluate(high)
    if high_eval["sample_error"] > epsilon:
        raise RuntimeError(f"failed to reach epsilon={epsilon} by r={r_max}")

    while low + 1 < high:
        mid = (low + high) // 2
        mid_eval = evaluate(mid)
        if mid_eval["sample_error"] <= epsilon:
            high = mid
            high_eval = mid_eval
        else:
            low = mid
    return high, high_eval


def main():
    args = parse_args()
    out_dir = Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)

    suffix = f"_{args.tag}" if args.tag else ""
    out_base = out_dir / (
        f"log_sampling_{args.sampling}_N{args.N}_T{args.T:g}_eps{args.epsilon:g}_"
        f"trials{args.trials}_repeats{args.repeats}_s0{args.s0 if args.s0 > 0 else 'auto'}{suffix}"
    )

    payload = {
        "script": "NCC_log_sampling.py",
        "params": {
            "N": args.N,
            "J": args.J,
            "h": args.h,
            "T": args.T,
            "epsilon": args.epsilon,
            "trials": args.trials,
            "repeats": args.repeats,
            "seed": args.seed,
            "r_max": args.r_max,
            "sampling": args.sampling,
            "s0": args.s0,
        },
        "r_mins": [],
        "sample_errors": [],
        "expectation_biases": [],
        "sample_fluctuations": [],
        "ci_low_history": [],
        "ci_high_history": [],
        "evaluations": [],
    }

    # Warm the cache before the first repetition so the expensive log objects are reused across r.
    build_log_static_data(
        args.N,
        args.epsilon,
        j=args.J,
        h=args.h,
        sampling=args.sampling,
        kappa=1,
        s0=args.s0 or None,
    )

    def checkpoint():
        save_checkpoint(out_base, payload)

    for repetition in range(args.repeats):
        label = f"[repeat {repetition + 1}/{args.repeats}]"
        r_min, metrics = find_r_min_sampling(
            n=args.N,
            t_total=args.T,
            epsilon=args.epsilon,
            trials=args.trials,
            repetition=repetition,
            base_seed=args.seed,
            r_max=args.r_max,
            j=args.J,
            h=args.h,
            sampling=args.sampling,
            s0=args.s0,
            evaluations=payload["evaluations"],
            progress_label=label,
            checkpoint_cb=checkpoint if args.save_every_eval else None,
        )
        payload["r_mins"].append(int(r_min))
        payload["sample_errors"].append(float(metrics["sample_error"]))
        payload["expectation_biases"].append(float(metrics["expectation_bias"]))
        payload["sample_fluctuations"].append(float(metrics["sample_fluctuation"]))
        _, ci_low, ci_high = confidence_interval(np.array(payload["r_mins"], dtype=float))
        payload["ci_low_history"].append(ci_low)
        payload["ci_high_history"].append(ci_high)
        save_checkpoint(out_base, payload)
        print(
            f"{label} r_min={r_min}, current mean={np.mean(payload['r_mins']):.3f}, "
            f"95% CI=[{ci_low:.3f}, {ci_high:.3f}]"
        )

    r_mins = np.array(payload["r_mins"], dtype=float)
    mean, ci_low, ci_high = confidence_interval(r_mins)
    print("finished search")
    print(f"r_min samples: {payload['r_mins']}")
    print(f"mean r_min: {mean:.3f}")
    print(f"95% CI for mean r_min: [{ci_low:.3f}, {ci_high:.3f}]")
    print(f"saved checkpoint to: {Path(f'{out_base}.json')}")
    print(f"saved arrays to: {Path(f'{out_base}.npz')}")


if __name__ == "__main__":
    main()
