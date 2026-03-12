import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

from NCC_log import build_static_data, build_tilde_V, cached_pauli_matrix_from_label


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
    parser.add_argument(
        "--q0",
        type=int,
        default=0,
        help="max BCH order kept in tilde_F construction (0 => ceil(log(2N/epsilon)))",
    )
    parser.add_argument("--save-every-search", action="store_true", help="checkpoint after every sampled r search step")
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


def make_search_seed(base_seed: int, repetition: int, r: int) -> int:
    return base_seed + repetition * 100_003 + r * 1_009


def confidence_interval(values: np.ndarray) -> tuple[float, float, float]:
    mean = float(np.mean(values))
    if len(values) <= 1:
        return mean, mean, mean
    std = float(np.std(values, ddof=1))
    half_width = 1.96 * std / math.sqrt(len(values))
    return mean, mean - half_width, mean + half_width


"""The following three functions handle the path"""


def resolve_output_dir(base_out_dir: Path, tag: str) -> Path:
    output_dir = base_out_dir / "smoke" if "smoke" in tag.lower() else base_out_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def normalized_tag_suffix(tag: str) -> str:
    if not tag:
        return ""
    if "smoke" in tag.lower():
        return "_smoke"
    return f"_{tag}"


def sampling_output_path(args, q0, s0) -> Path:
    suffix = normalized_tag_suffix(args.tag)
    output_dir = resolve_output_dir(args.out_dir, args.tag)
    return output_dir / (f"NCC_log_sampling_r_N{args.N}_T{args.T:g}_eps{args.epsilon:g}_" f"trials{args.trials}_repeats{args.repeats}_q{q0}_s{s0}{suffix}")


def grouped_search_array(payload: dict, key: str, fill_value, dtype):
    grouped: dict[int, list] = {}
    for item in payload["searches"]:
        grouped.setdefault(int(item["repetition"]), []).append(item[key])
    if not grouped:
        return np.empty((0, 0), dtype=dtype), np.empty((0,), dtype=int)
    reps = sorted(grouped)
    lengths = np.array([len(grouped[rep]) for rep in reps], dtype=int)
    width = int(lengths.max())
    array = np.full((len(reps), width), fill_value, dtype=dtype)
    for row, rep in enumerate(reps):
        values = np.array(grouped[rep], dtype=dtype)
        array[row, : len(values)] = values
    return array, lengths


def save_sampling_checkpoint(output_path: Path, payload: dict):
    json_path = Path(f"{output_path}.json")
    npz_path = Path(f"{output_path}.npz")
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    searched_r, searched_lengths = grouped_search_array(payload, "r", -1, int)
    searched_seed, _ = grouped_search_array(payload, "seed", -1, int)
    searched_sample_error, _ = grouped_search_array(payload, "sample_error", np.nan, float)
    searched_expectation_bias, _ = grouped_search_array(payload, "expectation_bias", np.nan, float)
    searched_sample_fluctuation, _ = grouped_search_array(payload, "sample_fluctuation", np.nan, float)
    np.savez(
        npz_path,
        sampled_r_mins=np.array(payload["sampled_r_mins"], dtype=int),
        expected_r_mins=np.array(payload["expected_r_mins"], dtype=int),
        sample_errors=np.array(payload["sample_errors"], dtype=float),
        expectation_biases=np.array(payload["expectation_biases"], dtype=float),
        sample_fluctuations=np.array(payload["sample_fluctuations"], dtype=float),
        low_bounds=np.array(payload["ci_low_history"], dtype=float),
        high_bounds=np.array(payload["ci_high_history"], dtype=float),
        searched_repetition=np.array([item["repetition"] for item in payload["searches"]], dtype=int),
        searched_lengths=searched_lengths,
        searched_r=searched_r,
        searched_seed=searched_seed,
        searched_sample_error=searched_sample_error,
        searched_expectation_bias=searched_expectation_bias,
        searched_sample_fluctuation=searched_sample_fluctuation,
    )


def sample_Pauli_then_compensate_exp(
    rng: np.random.Generator,
    identity: np.ndarray,
    F_terms: dict,
    order: int,
    eta_pair_sum: float,
    pair_scale: float,
    raw_total: float,
    atol: float = 1e-10,
):
    """Sample one compensated Pauli term for log-NCC."""
    data = F_terms[order]
    labels = data["labels"]
    probs = np.abs(data["coeffs"]) / data["l1_norm"]
    # sample Pauli from the expansion of given F_term
    idx = int(rng.choice(len(labels), p=probs))
    coeff = data["coeffs"][idx]
    pauli = cached_pauli_matrix_from_label(labels[idx])
    if data["kind"] == "pair":
        phase = coeff / (1j * abs(coeff))
        W_mat = phase * pauli
        hermitian_err = np.linalg.norm(W_mat - W_mat.conj().T, ord="fro")
        if hermitian_err > atol:
            raise ValueError(f"sampled W is not Hermitian (herm_err={hermitian_err:.3e})")
        # apply exp(i theta W), you must multiply total 1-norm raw_total.
        return raw_total * ((identity + 1j * eta_pair_sum * W_mat) / pair_scale)
    # apply W like (1+i)X, you must multiply total 1-norm raw_total.
    return raw_total * (coeff / abs(coeff) * pauli)


def estimate_total_sample_error(
    static: dict,
    t_total: float,
    r: int,
    trials: int,
    seed: int,
    evolution_data: dict,
    expectation_bias: float,
):
    """Estimate sampled total error for a fixed r using Monte Carlo trajectories."""
    s_orders = static["s_orders"]
    F_terms = static["F_terms"]
    identity = static["identity"]
    raw_weights = evolution_data["raw_weights"]
    raw_total = float(sum(raw_weights.values()))
    p_order = np.array([raw_weights[s] / raw_total for s in s_orders], dtype=float)

    rng = np.random.default_rng(seed)
    U_total_average = np.zeros_like(identity)
    for _ in tqdm(range(trials), desc=f"r={r}", leave=False, disable=not sys.stderr.isatty()):
        evo = identity.copy()
        for _ in range(r):
            order = int(rng.choice(s_orders, p=p_order))
            evo = (
                sample_Pauli_then_compensate_exp(
                    rng,
                    identity,
                    F_terms,
                    order,
                    evolution_data["eta_pair_sum"],
                    evolution_data["pair_scale"],
                    raw_total,
                )
                @ evolution_data["s1"]
                @ evo
            )
        U_total_average += evo
    U_total_average /= trials

    deterministic = np.linalg.matrix_power(evolution_data["tilde_V"] @ evolution_data["s1"], r)
    return {
        "sample_error": float(np.linalg.norm(U_total_average - evolution_data["U_exact"], 2)),
        "sample_fluctuation": float(np.linalg.norm(U_total_average - deterministic, 2)),
        "expectation_bias": expectation_bias,
        "eta_pair_sum": float(evolution_data["eta_pair_sum"]),
        "pair_scale": float(evolution_data["pair_scale"]),
        "raw_total": raw_total,
    }


"""Search start here"""


def search_r_min(
    static: dict,
    evolution_cache: dict[int, dict],
    t_total: float,
    epsilon: float,
    trials: int,
    repetition: int,
    base_seed: int,
    r_max: int,
    searches: list[dict],
    progress_label: str,
    checkpoint_cb=None,
):
    """Search expected and sampled r_min while reusing cached search results."""
    result_cache: dict[int, dict] = {}

    def search_min_by(metric_key: str) -> tuple[int, dict]:
        # Exponential search finds an upper bound; binary search then locates
        # the smallest r whose chosen metric is below epsilon.
        def evaluate_at_r(r: int) -> dict:
            # For expectation_bias, reuse deterministic evolution data only.
            # For sample_error, run the full Monte Carlo evaluation once and cache it.
            if metric_key == "expectation_bias":
                if r not in evolution_cache:
                    evolution_cache[r] = build_tilde_V(static, t_total, r)
                return {"expectation_bias": evolution_cache[r]["deterministic_bias"]}

            elif metric_key == "sample_error":
                if r in result_cache:
                    return result_cache[r]
                seed = make_search_seed(base_seed, repetition, r)
                if r not in evolution_cache:
                    evolution_cache[r] = build_tilde_V(static, t_total, r)
                evolution_data = evolution_cache[r]
                result = estimate_total_sample_error(
                    static=static,
                    t_total=t_total,
                    r=r,
                    trials=trials,
                    seed=seed,
                    evolution_data=evolution_data,
                    expectation_bias=evolution_data["deterministic_bias"],
                )
                result["seed"] = seed
                result_cache[r] = result
                searches.append(
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
            else:
                raise ValueError(f"unsupported metric_key={metric_key}")

        low = 0
        high = 1
        high_eval = evaluate_at_r(high)
        while high_eval[metric_key] > epsilon and high < r_max:
            low = high
            high *= 2
            high_eval = evaluate_at_r(high)
        if high_eval[metric_key] > epsilon:
            raise RuntimeError(f"failed to reach epsilon={epsilon} by r={r_max} for {metric_key}")

        while low + 1 < high:
            mid = (low + high) // 2
            mid_eval = evaluate_at_r(mid)
            if mid_eval[metric_key] <= epsilon:
                high = mid
                high_eval = mid_eval
            else:
                low = mid
        return high, high_eval

    # expected first to increase cache
    expected_r_min, _ = search_min_by("expectation_bias")
    sampled_r_min, sampled_eval = search_min_by("sample_error")
    return sampled_r_min, sampled_eval, expected_r_min


def main(argv=None):
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    s0 = int(np.ceil(np.log(4 / args.epsilon))) if args.s0 <= 0 else args.s0
    s0 = max(3, s0)
    q0 = int(np.ceil(np.log(2 * args.N / args.epsilon))) if args.q0 <= 0 else args.q0
    q0 = max(3, q0)
    output_path = sampling_output_path(args, q0, s0)
    static = build_static_data(n=args.N, q0=q0, s0=s0, epsilon=args.epsilon, j=args.J, h=args.h, K=1)
    evolution_cache: dict[int, dict] = {}
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
            "s0": s0,
            "q0": q0,
        },
        "sampled_r_mins": [],
        "expected_r_mins": [],
        "sample_errors": [],
        "expectation_biases": [],
        "sample_fluctuations": [],
        "ci_low_history": [],
        "ci_high_history": [],
        "searches": [],
    }

    def checkpoint():
        # Persist the partial payload so long runs can be resumed/reviewed even
        # if the process stops mid-search.
        save_sampling_checkpoint(output_path, payload)

    for repetition in range(args.repeats):
        label = f"[repeat {repetition + 1}/{args.repeats}]"
        sampled_r_min, metrics, expected_r_min = search_r_min(
            static=static,
            evolution_cache=evolution_cache,
            t_total=args.T,
            epsilon=args.epsilon,
            trials=args.trials,
            repetition=repetition,
            base_seed=args.seed,
            r_max=args.r_max,
            searches=payload["searches"],
            progress_label=label,
            checkpoint_cb=checkpoint if args.save_every_search else None,
        )
        payload["sampled_r_mins"].append(int(sampled_r_min))
        payload["expected_r_mins"].append(-1 if expected_r_min is None else int(expected_r_min))
        payload["sample_errors"].append(float(metrics["sample_error"]))
        payload["expectation_biases"].append(float(metrics["expectation_bias"]))
        payload["sample_fluctuations"].append(float(metrics["sample_fluctuation"]))
        _, ci_low, ci_high = confidence_interval(np.array(payload["sampled_r_mins"], dtype=float))
        payload["ci_low_history"].append(ci_low)
        payload["ci_high_history"].append(ci_high)
        save_sampling_checkpoint(output_path, payload)
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
    print(f"saved checkpoint to: {Path(f'{output_path}.json')}")
    print(f"saved arrays to: {Path(f'{output_path}.npz')}")


if __name__ == "__main__":
    main()
