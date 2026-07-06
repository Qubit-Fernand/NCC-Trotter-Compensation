#!/usr/bin/env python3
"""Run NCC_channel_log expected-r_min search with static/slim/full caches.

This local helper intentionally does not run Monte Carlo sampling. It evaluates
only the deterministic expectation-bias criterion once per case.
"""

from __future__ import annotations

import argparse
import gzip
import json
import pickle
import sys
from pathlib import Path

import numpy as np

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = next(
    (
        parent
        for parent in SCRIPT_PATH.parents
        if (parent / "scaling/NCC_channel_find_r_min.py").exists()
    ),
    SCRIPT_PATH.parents[1],
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scaling.NCC_channel_find_r_min as runner
import scaling.NCC_channel_log as ncc_channel_log


FULL_BATCH_FIELDS = {
    "basis_states",
    "basis_labels",
    "exact_output_batch",
    "deterministic_output_batch",
}


def first_existing(*paths: Path) -> Path:
    for path in paths:
        if path.exists():
            return path
    return paths[0]


DEFAULT_STATIC_CACHE = first_existing(
    REPO_ROOT / ".bohrium.cache/static_cache.pkl.gz",
    REPO_ROOT
    / ".bohrium/active/log_N8_eps_split_cache_20260705/cache_assets/static_cache_NCC_channel_log_N8_q4_s4_J1_h1_K1.pkl.gz",
)
DEFAULT_SLIM_CACHE = first_existing(
    REPO_ROOT / ".bohrium.cache/slim_evolution_cache.pkl.gz",
    REPO_ROOT
    / ".bohrium/active/log_N8_eps_split_cache_20260705/cache_assets/evolution_cache_NCC_channel_log_N8_T2_pid3279512.pkl.gz",
)
FINAL_FULL_CACHE = REPO_ROOT / ".bohrium.cache/full_evolution_cache.pkl.gz"
if FINAL_FULL_CACHE.exists():
    DEFAULT_FULL_CACHES = [FINAL_FULL_CACHE]
else:
    DEFAULT_FULL_CACHES = [
        path
        for path in [
            REPO_ROOT
            / ".bohrium/log_N8_eps_split_cache_20260705/cache_assets/evolution_cache_FULL_NCC_channel_log_N8_T2_powers_of_two.pkl.gz",
            REPO_ROOT
            / ".bohrium/active/log_N8_eps_split_cache_20260705/cache_assets/evolution_cache_FULL_NCC_channel_log_N8_T2_r24_48_20260706.pkl.gz",
        ]
        if path.exists()
    ]


def load_pickle(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as handle:
        return pickle.load(handle)


def normalize_static_key(key) -> tuple[int, int, int, float, float, int]:
    n, q0, s0, j, h, k = key
    return int(n), int(q0), int(s0), float(j), float(h), int(k)


def normalize_evolution_key(key) -> tuple[int, float, int]:
    n, t_total, r = key
    return int(n), float(t_total), int(r)


def hydrate_log_evolution_data(static: dict, evolution_data: dict) -> dict:
    """Reattach non-pickleable NCC_channel_log evolution callables."""
    if "apply_tilde_V_expectation" in evolution_data:
        return evolution_data

    F_terms = static["F_terms"]
    s_orders = static["s_orders"]
    raw_weights = evolution_data["raw_weights"]
    pair_components = evolution_data["pair_data"]
    eta_pair_sum = evolution_data["eta_pair_sum"]
    pair_scale = evolution_data["pair_scale"]
    t = evolution_data["t"]

    def apply_tail_F_channel(order: int, rho):
        rho_batch, single_input = ncc_channel_log._as_density_matrix_batch(rho)
        out = rho_batch * 0
        for phi_tuple in F_terms[order]["phi_tuples"]:
            tuple_out = rho_batch
            for phi_order in reversed(phi_tuple["q_tuple"]):
                tuple_out = ncc_channel_log.apply_ad_commutator(static["Phi_terms"][phi_order], tuple_out)
            out += phi_tuple["factorial_j_scale"] * tuple_out
        return ncc_channel_log._restore_density_matrix_shape(out, single_input)

    def apply_tilde_V_taylor(rho):
        rho_batch, single_input = ncc_channel_log._as_density_matrix_batch(rho)
        out = rho_batch.copy()
        for order in s_orders:
            if F_terms[order]["kind"] == "pair":
                out += (t**order) * ncc_channel_log.apply_ad_commutator(
                    static["tilde_F_operator_terms"][order],
                    rho_batch,
                )
            else:
                out += (t**order) * apply_tail_F_channel(order, rho_batch)
        return ncc_channel_log._restore_density_matrix_shape(out, single_input)

    def apply_tilde_V_expectation(rho):
        rho_batch, single_input = ncc_channel_log._as_density_matrix_batch(rho)
        out = rho_batch * 0
        for order in s_orders:
            weight = raw_weights[order]
            data = F_terms[order]
            if data["kind"] == "pair":
                for pauli in pair_components[order]:
                    out += (
                        weight
                        * pauli["prob"]
                        * (
                            rho_batch
                            + eta_pair_sum
                            * ncc_channel_log.apply_ad_commutator(1j * pauli["W_mat"], rho_batch)
                        )
                        / pair_scale
                    )
            else:
                out += weight * apply_tail_F_channel(order, rho_batch) / data["sampling_l1_norm"]
        return ncc_channel_log._restore_density_matrix_shape(out, single_input)

    evolution_data["apply_tilde_V_taylor"] = apply_tilde_V_taylor
    evolution_data["apply_tilde_V_expectation"] = apply_tilde_V_expectation
    return evolution_data


def install_static_cache(path: Path, required: bool) -> dict:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"required static cache not found: {path}")
        print(f"[cache] static cache missing, continuing without it: {path}", flush=True)
        return {}

    raw_cache = load_pickle(path)
    if not isinstance(raw_cache, dict):
        raise TypeError(f"static cache must be a dict, got {type(raw_cache)!r}")
    static_cache = {normalize_static_key(key): value for key, value in raw_cache.items()}
    original_build_static_data = ncc_channel_log.build_static_data

    def cached_build_static_data(n, q0, s0, j=1.0, h=1.0, K=1):
        key = (int(n), int(q0), int(s0), float(j), float(h), int(K))
        if key in static_cache:
            print(f"[cache] static hit {key}", flush=True)
            return static_cache[key]
        print(f"[cache] static miss {key}; building locally", flush=True)
        return original_build_static_data(n=n, q0=q0, s0=s0, j=j, h=h, K=K)

    ncc_channel_log.build_static_data = cached_build_static_data
    runner.ncc_channel_log.build_static_data = cached_build_static_data
    print(f"[cache] loaded {len(static_cache)} static entries from {path}", flush=True)
    return static_cache


def pick_static_for_evolution(static_cache: dict, evolution_key: tuple[int, float, int]):
    for static_key, static in static_cache.items():
        if static_key[0] == evolution_key[0]:
            return static
    return None


def install_evolution_cache(path: Path, static_cache: dict, required: bool) -> None:
    if not path.exists():
        if required:
            raise FileNotFoundError(f"required evolution cache not found: {path}")
        print(f"[cache] evolution cache missing, continuing without it: {path}", flush=True)
        return

    raw_cache = load_pickle(path)
    if not isinstance(raw_cache, dict):
        raise TypeError(f"evolution cache must be a dict, got {type(raw_cache)!r}")

    loaded = 0
    overwritten = 0
    full_entries = 0
    for key, value in raw_cache.items():
        norm_key = normalize_evolution_key(key)
        static = pick_static_for_evolution(static_cache, norm_key)
        if static is not None:
            value = hydrate_log_evolution_data(static, value)
        if norm_key in runner.EVOLUTION_CACHE:
            overwritten += 1
        if FULL_BATCH_FIELDS <= set(value):
            full_entries += 1
        runner.EVOLUTION_CACHE[norm_key] = value
        loaded += 1

    print(
        f"[cache] loaded {loaded} evolution entries from {path} "
        f"(overwrote {overwritten}, full entries {full_entries})",
        flush=True,
    )


def expected_output_path(args, q0: int, s0: int) -> Path:
    output_dir = runner.resolve_output_dir(args.out_dir, args.tag)
    suffix = runner.normalized_tag_suffix(args.tag)
    stem = (
        f"NCC_channel_log_expected_r_min_N{args.N}_T{args.T:g}_eps{args.epsilon:g}"
        f"_q{q0}_s{s0}"
    )
    return output_dir / f"{stem}{suffix}"


def save_expected_checkpoint(output_path: Path, payload: dict) -> None:
    json_path = Path(f"{output_path}.json")
    npz_path = Path(f"{output_path}.npz")
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    searches = payload["searches"]
    np.savez(
        npz_path,
        searched_r=np.array([item["r"] for item in searches], dtype=int),
        searched_expectation_bias=np.array([item["expectation_bias"] for item in searches], dtype=float),
        expected_r_min=np.array([payload["expected_r_min"]], dtype=int),
        expectation_bias_at_min=np.array([payload["expectation_bias_at_min"]], dtype=float),
    )


def ensure_expected_batches(mode_impl, static: dict, evolution_data: dict, r: int):
    missing = sorted(FULL_BATCH_FIELDS - set(evolution_data))
    if missing:
        key = (int(static["n"]), float(evolution_data.get("t", np.nan)) * int(r), int(r))
        print(
            f"[cache] building full batch r={int(r)} key={key} missing={','.join(missing)}",
            flush=True,
        )
    return runner.ensure_basis_batches(mode_impl, static, evolution_data, r)


def search_expected_r_min(
    static: dict,
    t_total: float,
    epsilon: float,
    r_max: int,
    searches: list[dict],
    checkpoint_cb=None,
    mode_impl=None,
) -> tuple[int, dict]:
    result_cache: dict[int, dict] = {}

    def evaluate_at_r(r: int) -> dict:
        if r in result_cache:
            return result_cache[r]
        cache_key = (int(static["n"]), float(t_total), int(r))
        if cache_key not in runner.EVOLUTION_CACHE:
            print(f"[cache] building tilde_V r={int(r)} key={cache_key}", flush=True)
            runner.EVOLUTION_CACHE[cache_key] = mode_impl.build_tilde_V(static, t_total, r)
        evolution_data = runner.EVOLUTION_CACHE[cache_key]
        exact_output_batch, deterministic_output_batch = ensure_expected_batches(
            mode_impl,
            static,
            evolution_data,
            r,
        )
        expectation_bias, worst_index = runner.max_trace_norm_error(
            deterministic_output_batch,
            exact_output_batch,
            threshold=epsilon,
        )
        result = {
            "r": int(r),
            "expectation_bias": float(expectation_bias),
            "worst_index": -1 if worst_index is None else int(worst_index),
            "eta_pair_sum": float(evolution_data["eta_pair_sum"]),
            "pair_scale": float(evolution_data["pair_scale"]),
            "raw_l1_norm_total": float(evolution_data["raw_l1_norm_total"]),
        }
        result_cache[r] = result
        searches.append(result)
        print(
            f"[expected] eval r={r}: bias={result['expectation_bias']:.6e}, "
            f"threshold={epsilon:.6e}",
            flush=True,
        )
        if checkpoint_cb is not None:
            checkpoint_cb()
        return result

    low = 0
    high = 1
    high_eval = evaluate_at_r(high)
    while high_eval["expectation_bias"] > epsilon and high < r_max:
        low = high
        high *= 2
        high_eval = evaluate_at_r(high)
    if high_eval["expectation_bias"] > epsilon:
        raise RuntimeError(f"failed to reach epsilon={epsilon} by r={r_max} for expectation_bias")

    while low + 1 < high:
        mid = (low + high) // 2
        mid_eval = evaluate_at_r(mid)
        if mid_eval["expectation_bias"] <= epsilon:
            high = mid
            high_eval = mid_eval
        else:
            low = mid
    return high, high_eval


def run_case(args, case_index: int, total_cases: int) -> Path:
    if args.mode != "log":
        raise ValueError(f"expected-only cached helper supports mode='log', got {args.mode!r}")

    s0 = int(np.ceil(np.log(4 / args.epsilon))) if args.s0 <= 0 else args.s0
    s0 = max(3, s0)
    q0 = int(np.ceil(np.log(2 * args.N / args.epsilon))) if args.q0 <= 0 else args.q0
    q0 = max(3, q0)

    output_path = expected_output_path(args, q0, s0)
    static = ncc_channel_log.build_static_data(n=args.N, q0=q0, s0=s0, j=args.J, h=args.h, K=1)
    payload = {
        "script": "NCC_channel_expected_r_min_cached.py",
        "mode": args.mode,
        "params": {
            "N": args.N,
            "J": args.J,
            "h": args.h,
            "T": args.T,
            "epsilon": args.epsilon,
            "requested_repeats": args.repeats,
            "base_seed": args.seed,
            "r_max": args.r_max,
            "s0": s0,
            "q0": q0,
        },
        "expected_r_min": None,
        "expectation_bias_at_min": None,
        "searches": [],
    }

    def checkpoint():
        if payload["expected_r_min"] is not None:
            save_expected_checkpoint(output_path, payload)
        else:
            Path(f"{output_path}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))

    print(f"[case {case_index}/{total_cases}] expected-only search start", flush=True)
    expected_r_min, metrics = search_expected_r_min(
        static=static,
        t_total=args.T,
        epsilon=args.epsilon,
        r_max=args.r_max,
        searches=payload["searches"],
        checkpoint_cb=checkpoint if args.save_every_search else None,
        mode_impl=ncc_channel_log,
    )
    payload["expected_r_min"] = int(expected_r_min)
    payload["expectation_bias_at_min"] = float(metrics["expectation_bias"])
    save_expected_checkpoint(output_path, payload)
    print(
        f"[case {case_index}/{total_cases}] expected_r_min={expected_r_min}, "
        f"bias={metrics['expectation_bias']:.6e}",
        flush=True,
    )
    print(f"[case {case_index}/{total_cases}] saved checkpoint to: {output_path}.json", flush=True)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-file", type=Path, required=True)
    parser.add_argument("--static-cache", type=Path, default=DEFAULT_STATIC_CACHE)
    parser.add_argument("--slim-evolution-cache", type=Path, default=DEFAULT_SLIM_CACHE)
    parser.add_argument(
        "--full-evolution-cache",
        type=Path,
        action="append",
        default=None,
        help="Full evolution cache layer. May be passed multiple times.",
    )
    parser.add_argument("--require-static-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--require-evolution-cache", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    static_cache = install_static_cache(args.static_cache, args.require_static_cache)
    install_evolution_cache(args.slim_evolution_cache, static_cache, args.require_evolution_cache)
    full_evolution_caches = args.full_evolution_cache
    if full_evolution_caches is None:
        full_evolution_caches = DEFAULT_FULL_CACHES
    if args.require_evolution_cache and not full_evolution_caches:
        raise FileNotFoundError("no full evolution cache found; pass --full-evolution-cache explicitly")
    for path in full_evolution_caches:
        install_evolution_cache(path, static_cache, args.require_evolution_cache)
    print(f"[cache] final EVOLUTION_CACHE entries: {len(runner.EVOLUTION_CACHE)}", flush=True)

    cases = runner.load_batch_cases(args.batch_file)
    print(f"loaded {len(cases)} cases from {args.batch_file}", flush=True)
    for idx, case in enumerate(cases, start=1):
        run_case(runner.case_namespace(case), case_index=idx, total_cases=len(cases))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
