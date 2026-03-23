import argparse
import json
import math
from copy import deepcopy
from pathlib import Path

import numpy as np
from tqdm import tqdm

import NCC_log as ncc_log
import NCC_original as ncc_original
from Pauli_Hamiltonian_BCH import cached_pauli_matrix_from_label


def build_parser():
    parser = argparse.ArgumentParser(description="Unified sampling-based r_min search for original/log NCC.")
    parser.add_argument("--mode", choices=("original", "log"), default="log", help="which NCC variant to run")
    parser.add_argument("--out-dir", type=Path, default=Path("data"), help="output directory")
    parser.add_argument("--tag", type=str, default="", help="optional suffix for output file names")
    parser.add_argument(
        "--batch-file",
        type=Path,
        default=None,
        help="optional JSON file listing multiple case overrides to run sequentially in one Python process",
    )
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
    parser.add_argument("--q0", type=int, default=0, help="override BCH truncation order")
    parser.add_argument("--save-every-search", action="store_true", help="checkpoint after every sampled r search step")
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

        normalized = dict(defaults)
        normalized.update(case)
        if "out_dir" in normalized:
            normalized["out_dir"] = Path(normalized["out_dir"])
        normalized_cases.append(normalized)
    return normalized_cases


# copy the (default) args, override with the case-specific values, and return a new Namespace for the case
def case_args_from_overrides(base_args, overrides: dict):
    case_args = argparse.Namespace(**vars(deepcopy(base_args)))
    for key, value in overrides.items():
        setattr(case_args, key, value)
    return case_args


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


def sample_original_pauli_then_compensate_exp(
    rng: np.random.Generator,
    static: dict,
    p_s: np.ndarray,
    eta_sum: float,
    atol: float = 1e-10,
) -> np.ndarray:
    """Sample one compensated Pauli term for original-NCC."""
    order = int(rng.choice([2, 3], p=p_s))
    if order == 2:
        coeffs = static["c1_coeffs"]
        labels = static["c1_labels"]
        l1_norm = static["c1_l1"]
    elif order == 3:
        coeffs = static["c2_coeffs"]
        labels = static["c2_labels"]
        l1_norm = static["c2_l1"]
    else:
        raise ValueError(f"unsupported sampled order={order}")
    probs = np.abs(coeffs) / l1_norm
    idx = int(rng.choice(len(labels), p=probs))
    coeff = coeffs[idx]
    sign = coeff / (1j * abs(coeff))
    w_mat = sign * cached_pauli_matrix_from_label(labels[idx])
    hermitian_err = np.linalg.norm(w_mat - w_mat.conj().T, ord="fro")
    if hermitian_err > atol:
        raise ValueError(f"sampled W is not Hermitian (herm_err={hermitian_err:.3e})")
    return static["identity"] + 1j * eta_sum * w_mat


def estimate_total_sample_error_original(
    static: dict,
    t_total: float,
    r: int,
    trials: int,
    seed: int,
    evolution_data: dict,
    expectation_bias: float,
):
    """Estimate sampled total error for original-NCC."""
    del t_total
    rng = np.random.default_rng(seed)
    u_total_average = np.zeros_like(static["identity"])
    for _ in tqdm(range(trials), desc=f"r={r}", leave=False):
        evo = static["identity"].copy()
        for _ in range(r):
            evo = (
                sample_original_pauli_then_compensate_exp(
                    rng,
                    static,
                    evolution_data["p_s"],
                    evolution_data["eta_sum"],
                )
                @ evolution_data["S1"]
                @ evo
            )
        u_total_average += evo
    u_total_average /= trials
    return {
        "sample_error": float(np.linalg.norm(u_total_average - evolution_data["U_exact"], 2)),
        "sample_fluctuation": float(np.linalg.norm(u_total_average - evolution_data["deterministic"], 2)),
        "expectation_bias": expectation_bias,
    }


def sample_log_pauli_then_compensate_exp(
    rng: np.random.Generator,
    identity: np.ndarray,
    f_terms: dict,
    order: int,
    eta_pair_sum: float,
    pair_scale: float,
    raw_total: float,
    atol: float = 1e-10,
):
    """Sample one compensated Pauli term for log-NCC."""
    data = f_terms[order]
    labels = data["labels"]
    probs = np.abs(data["coeffs"]) / data["l1_norm"]
    idx = int(rng.choice(len(labels), p=probs))
    coeff = data["coeffs"][idx]
    pauli = cached_pauli_matrix_from_label(labels[idx])
    if data["kind"] == "pair":
        phase = coeff / (1j * abs(coeff))
        w_mat = phase * pauli
        hermitian_err = np.linalg.norm(w_mat - w_mat.conj().T, ord="fro")
        if hermitian_err > atol:
            raise ValueError(f"sampled W is not Hermitian (herm_err={hermitian_err:.3e})")
        return raw_total * ((identity + 1j * eta_pair_sum * w_mat) / pair_scale)
    return raw_total * (coeff / abs(coeff) * pauli)


def estimate_total_sample_error_log(
    static: dict,
    t_total: float,
    r: int,
    trials: int,
    seed: int,
    evolution_data: dict,
    expectation_bias: float,
):
    """Estimate sampled total error for log-NCC."""
    del t_total
    s_orders = static["s_orders"]
    f_terms = static["F_terms"]
    identity = static["identity"]
    raw_weights = evolution_data["raw_weights"]
    raw_total = float(sum(raw_weights.values()))
    p_order = np.array([raw_weights[s] / raw_total for s in s_orders], dtype=float)

    rng = np.random.default_rng(seed)
    u_total_average = np.zeros_like(identity)
    for _ in tqdm(range(trials), desc=f"r={r}", leave=False):
        evo = identity.copy()
        for _ in range(r):
            order = int(rng.choice(s_orders, p=p_order))
            evo = (
                sample_log_pauli_then_compensate_exp(
                    rng,
                    identity,
                    f_terms,
                    order,
                    evolution_data["eta_pair_sum"],
                    evolution_data["pair_scale"],
                    raw_total,
                )
                @ evolution_data["S1"]
                @ evo
            )
        u_total_average += evo
    u_total_average /= trials
    deterministic = np.linalg.matrix_power(evolution_data["tilde_V"] @ evolution_data["S1"], r)
    return {
        "sample_error": float(np.linalg.norm(u_total_average - evolution_data["U_exact"], 2)),
        "sample_fluctuation": float(np.linalg.norm(u_total_average - deterministic, 2)),
        "expectation_bias": expectation_bias,
    }


def build_mode_config(args):
    if args.mode == "log":
        s0 = int(np.ceil(np.log(4 / args.epsilon))) if args.s0 <= 0 else args.s0
        s0 = max(3, s0)
        q0 = int(np.ceil(np.log(2 * args.N / args.epsilon))) if args.q0 <= 0 else args.q0
        q0 = max(3, q0)
        static = ncc_log.build_static_data(n=args.N, q0=q0, s0=s0, epsilon=args.epsilon, j=args.J, h=args.h, K=1)
        return {
            "static": static,
            "q0": q0,
            "s0": s0,
            "build_tilde_V": ncc_log.build_tilde_V,
            "estimate_total_sample_error": estimate_total_sample_error_log,
        }

    static = ncc_original.build_static_data(args.N, args.J, args.h)
    return {
        "static": static,
        "q0": None,
        "s0": None,
        "build_tilde_V": ncc_original.build_tilde_V,
        "estimate_total_sample_error": estimate_total_sample_error_original,
    }


def search_r_min(
    static: dict,
    build_tilde_V_fn,
    estimate_total_sample_error_fn,
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
                evolution_cache[r] = build_tilde_V_fn(static, t_total, r)

            if metric_key == "expectation_bias":
                return {"expectation_bias": evolution_cache[r]["deterministic_bias"]}

            if metric_key == "sample_error":
                if r in result_cache:
                    return result_cache[r]
                seed = make_search_seed(base_seed, repetition, r)
                evolution_data = evolution_cache[r]
                result = estimate_total_sample_error_fn(
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
            build_tilde_V_fn=config["build_tilde_V"],
            estimate_total_sample_error_fn=config["estimate_total_sample_error"],
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
    if args.batch_file is None:
        run_case(args)
        return

    cases = load_batch_cases(args.batch_file)
    print(f"loaded {len(cases)} cases from {args.batch_file}")
    for idx, overrides in enumerate(cases, start=1):
        case_args = case_args_from_overrides(args, overrides)
        case_args.batch_file = None
        run_case(case_args, case_index=idx, total_cases=len(cases))


if __name__ == "__main__":
    main()
