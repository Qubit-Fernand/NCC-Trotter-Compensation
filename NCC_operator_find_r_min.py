"""
Too small r cause large step t, the BCH not convergent,
tilde_V is far from unitary, norm maybe very large.
"""

import argparse
import json
import math
from pathlib import Path

import numpy as np
from tqdm import tqdm

import NCC_operator_log as ncc_log
import NCC_operator_original as ncc_original


def build_parser():
    parser = argparse.ArgumentParser(description="Run unified sampling-based r_min searches from a JSON batch file.")
    parser.add_argument(
        "--batch-file",
        type=Path,
        default=Path("run.json"),
        help="JSON file listing defaults and cases to run sequentially in one Python process",
    )
    return parser


def parse_args(argv=None):
    return build_parser().parse_args(argv)


CASE_FIELDS = {
    "mode",
    "out_dir",
    "tag",
    "N",
    "J",
    "h",
    "T",
    "epsilon",
    "trials",
    "repeats",
    "seed",
    "r_max",
    "s0",
    "q0",
    "save_every_search",
}

CASE_DEFAULTS = {
    "out_dir": Path("data"),
    "tag": "",
    "J": 1.0,
    "h": 1.0,
    "trials": 1000,
    "repeats": 10,
    "seed": 7,
    "r_max": 1024,
    "s0": 0,
    "q0": 0,
    "save_every_search": False,
}

REQUIRED_CASE_FIELDS = {"mode", "N", "T", "epsilon"}


def make_search_seed(base_seed: int, repetition: int, r: int) -> int:
    return base_seed + repetition * 100_003 + r * 1_009


def confidence_interval(values: np.ndarray) -> tuple[float, float, float]:
    mean = float(np.mean(values))
    if len(values) <= 1:
        return mean, mean, mean
    std = float(np.std(values, ddof=1))
    half_width = 1.96 * std / math.sqrt(len(values))
    return mean, mean - half_width, mean + half_width


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


def sampling_output_path(args, q0=None, s0=None) -> Path:
    suffix = normalized_tag_suffix(args.tag)
    output_dir = resolve_output_dir(args.out_dir, args.tag)
    if args.mode == "log":
        stem = f"NCC_log_sampling_r_N{args.N}_T{args.T:g}_eps{args.epsilon:g}_trials{args.trials}_repeats{args.repeats}_q{q0}_s{s0}"
    else:
        stem = f"NCC_original_sampling_r_N{args.N}_T{args.T:g}_eps{args.epsilon:g}_trials{args.trials}_repeats{args.repeats}"
    return output_dir / f"{stem}{suffix}"


def load_batch_cases(batch_file: Path) -> list[dict]:
    payload = json.loads(batch_file.read_text())
    defaults = {}
    if isinstance(payload, dict):
        defaults = payload.get("defaults", {})
        cases = payload.get("cases")
    else:
        cases = payload
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"batch file {batch_file} must contain a non-empty case list")
    if not isinstance(defaults, dict):
        raise ValueError(f"batch file {batch_file} has non-object defaults")

    default_unknown_keys = sorted(set(defaults) - CASE_FIELDS)
    if default_unknown_keys:
        raise ValueError(f"batch file {batch_file} has unsupported default keys: {', '.join(default_unknown_keys)}")

    normalized_cases = []
    for idx, case in enumerate(cases, start=1):
        if not isinstance(case, dict):
            raise ValueError(f"batch case #{idx} in {batch_file} must be an object")
        unknown_keys = sorted(set(case) - CASE_FIELDS)
        if unknown_keys:
            raise ValueError(f"batch case #{idx} in {batch_file} has unsupported keys: {', '.join(unknown_keys)}")

        normalized = dict(CASE_DEFAULTS)
        normalized.update(defaults)
        normalized.update(case)
        missing_keys = sorted(field for field in REQUIRED_CASE_FIELDS if field not in normalized)
        if missing_keys:
            raise ValueError(f"batch case #{idx} in {batch_file} is missing required keys: {', '.join(missing_keys)}")

        normalized["out_dir"] = Path(normalized["out_dir"])
        normalized_cases.append(normalized)
    return normalized_cases


def case_namespace(case: dict):
    return argparse.Namespace(**case)


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


def estimate_total_sample_error(
    static: dict,
    t_total: float,
    r: int,
    trials: int,
    seed: int,
    evolution_data: dict,
    expectation_bias: float,
    mode_impl,
):
    """Estimate sampled total error for the selected operator NCC mode."""
    del t_total

    rng = np.random.default_rng(seed)
    u_total_average = np.zeros_like(static["identity"])
    for _ in tqdm(range(trials), desc=f"r={r}", leave=False):
        evo = static["identity"].copy()
        for _ in range(r):
            evo = mode_impl.sample_Pauli_then_compensate_exp(rng, static, evolution_data) @ evolution_data["S1"] @ evo
        u_total_average += evo
    u_total_average /= trials
    return {
        "sample_error": float(np.linalg.norm(u_total_average - evolution_data["U_exact"], 2)),
        "sample_fluctuation": float(np.linalg.norm(u_total_average - evolution_data["deterministic"], 2)),
        "expectation_bias": expectation_bias,
    }


def build_mode_config(args):
    if args.mode == "log":
        s0 = int(np.ceil(np.log(4 / args.epsilon))) if args.s0 <= 0 else args.s0
        s0 = max(3, s0)
        q0 = int(np.ceil(np.log(2 * args.N / args.epsilon))) if args.q0 <= 0 else args.q0
        q0 = max(3, q0)
        static = ncc_log.build_static_data(n=args.N, q0=q0, s0=s0, j=args.J, h=args.h, K=1)
        return {
            "static": static,
            "q0": q0,
            "s0": s0,
            "mode_impl": ncc_log,
        }

    static = ncc_original.build_static_data(args.N, args.J, args.h)
    return {
        "static": static,
        "q0": None,
        "s0": None,
        "mode_impl": ncc_original,
    }


def search_r_min(
    static: dict,
    mode_impl,
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
    result_cache: dict[int, dict] = {}

    def search_min_by(metric_key: str) -> tuple[int, dict]:
        def evaluate_at_r(r: int) -> dict:
            if r not in evolution_cache:
                evolution_cache[r] = mode_impl.build_tilde_V(static, t_total, r)

            if metric_key == "expectation_bias":
                return {"expectation_bias": evolution_cache[r]["deterministic_bias"]}

            if metric_key == "sample_error":
                if r in result_cache:
                    return result_cache[r]
                seed = make_search_seed(base_seed, repetition, r)
                evolution_data = evolution_cache[r]
                result = estimate_total_sample_error(
                    static=static,
                    t_total=t_total,
                    r=r,
                    trials=trials,
                    seed=seed,
                    evolution_data=evolution_data,
                    expectation_bias=evolution_data["deterministic_bias"],
                    mode_impl=mode_impl,
                )
                result["seed"] = seed
                result_cache[r] = result
                searches.append({"repetition": repetition, "r": r, **result})
                print(
                    f"{progress_label} eval r={r}: sample_error={result['sample_error']:.6e}, "
                    f"bias={result['expectation_bias']:.6e}, fluct={result['sample_fluctuation']:.6e}"
                )
                if checkpoint_cb is not None:
                    checkpoint_cb()
                return result

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

    expected_r_min, _ = search_min_by("expectation_bias")
    sampled_r_min, sampled_eval = search_min_by("sample_error")
    return sampled_r_min, sampled_eval, expected_r_min


def run_case(args, case_index: int | None = None, total_cases: int | None = None):
    args.out_dir.mkdir(parents=True, exist_ok=True)
    config = build_mode_config(args)
    output_path = sampling_output_path(args, config["q0"], config["s0"])

    case_prefix = ""
    if case_index is not None and total_cases is not None:
        case_prefix = f"[case {case_index}/{total_cases}] "
    print(f"{case_prefix}starting mode={args.mode} N={args.N} T={args.T:g} epsilon={args.epsilon:g} " f"trials={args.trials} repeats={args.repeats}")

    payload = {
        "script": "NCC_sampling_r.py",
        "mode": args.mode,
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
            "q0": config["q0"],
            "s0": config["s0"],
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

    evolution_cache: dict[int, dict] = {}

    def checkpoint():
        save_sampling_checkpoint(output_path, payload)

    for repetition in range(args.repeats):
        label = f"[{args.mode} repeat {repetition + 1}/{args.repeats}]"
        sampled_r_min, metrics, expected_r_min = search_r_min(
            static=config["static"],
            mode_impl=config["mode_impl"],
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
        payload["expected_r_mins"].append(int(expected_r_min))
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
    print(f"{case_prefix}finished search")
    print(f"{case_prefix}mode: {args.mode}")
    print(f"sampled r_min samples: {payload['sampled_r_mins']}")
    print(f"expected r_min samples: {payload['expected_r_mins']}")
    print(f"sampled r_min mean: {mean:.3f}")
    print(f"95% CI for sampled r_min mean: [{ci_low:.3f}, {ci_high:.3f}]")
    print(f"saved checkpoint to: {Path(f'{output_path}.json')}")
    print(f"saved arrays to: {Path(f'{output_path}.npz')}")
    return output_path


def main(argv=None):
    args = parse_args(argv)
    cases = load_batch_cases(args.batch_file)
    print(f"loaded {len(cases)} cases from {args.batch_file}")
    for idx, case in enumerate(cases, start=1):
        run_case(case_namespace(case), case_index=idx, total_cases=len(cases))


if __name__ == "__main__":
    main()
