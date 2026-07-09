#!/usr/bin/env python3
"""Warm-start exact expected-r_min search for NCC_channel_log.

This keeps the exact integer-r evaluator from NCC_channel_expected_r_min_cached
but seeds the search bracket from an existing expected-r JSON file. It is useful
when a nearby epsilon has already evaluated candidate r values.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import NCC_channel_expected_r_min_cached as base


def save_checkpoint(output_path: Path, payload: dict) -> None:
    json_path = Path(f"{output_path}.json")
    npz_path = Path(f"{output_path}.npz")
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    if payload["expected_r_min"] is None:
        return
    searches = payload["searches"]
    np.savez(
        npz_path,
        searched_r=np.array([item["r"] for item in searches], dtype=int),
        searched_expectation_bias=np.array([item["expectation_bias"] for item in searches], dtype=float),
        expected_r_min=np.array([payload["expected_r_min"]], dtype=int),
        expectation_bias_at_min=np.array([payload["expectation_bias_at_min"]], dtype=float),
    )


def dedup_searches(searches: list[dict]) -> list[dict]:
    by_r = {}
    for item in searches:
        by_r[int(item["r"])] = dict(item, r=int(item["r"]))
    return [by_r[r] for r in sorted(by_r)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epsilon", type=float, required=True)
    parser.add_argument("--seed-json", type=Path, required=True)
    parser.add_argument("--static-cache", type=Path, default=base.DEFAULT_STATIC_CACHE)
    parser.add_argument("--slim-evolution-cache", type=Path, default=base.DEFAULT_SLIM_CACHE)
    parser.add_argument("--full-evolution-cache", type=Path, action="append", default=None)
    parser.add_argument("--N", type=int, default=8)
    parser.add_argument("--T", type=float, default=2.0)
    parser.add_argument("--J", type=float, default=1.0)
    parser.add_argument("--h", type=float, default=1.0)
    parser.add_argument("--q0", type=int, default=4)
    parser.add_argument("--s0", type=int, default=4)
    parser.add_argument("--r-max", type=int, default=4096)
    parser.add_argument("--out-dir", type=Path, default=Path("scaling/data"))
    args = parser.parse_args()

    static_cache = base.install_static_cache(args.static_cache, required=True)
    base.install_evolution_cache(args.slim_evolution_cache, static_cache, required=True)
    full_caches = args.full_evolution_cache if args.full_evolution_cache is not None else base.DEFAULT_FULL_CACHES
    for path in full_caches:
        base.install_evolution_cache(path, static_cache, required=True)
    print(f"[cache] final EVOLUTION_CACHE entries: {len(base.runner.EVOLUTION_CACHE)}", flush=True)

    seed_payload = json.loads(args.seed_json.read_text())
    seed_searches = dedup_searches(seed_payload.get("searches", []))
    searches = [dict(item) for item in seed_searches]
    result_cache = {int(item["r"]): item for item in searches}

    static = base.ncc_channel_log.build_static_data(n=args.N, q0=args.q0, s0=args.s0, j=args.J, h=args.h, K=1)
    output_path = args.out_dir / (
        f"NCC_channel_log_expected_r_min_N{args.N}_T{args.T:g}_eps{args.epsilon:g}_q{args.q0}_s{args.s0}"
    )
    payload = {
        "script": Path(__file__).name,
        "mode": "log",
        "params": {
            "N": args.N,
            "J": args.J,
            "h": args.h,
            "T": args.T,
            "epsilon": args.epsilon,
            "requested_repeats": 10,
            "base_seed": 7,
            "r_max": args.r_max,
            "s0": args.s0,
            "q0": args.q0,
        },
        "expected_r_min": None,
        "expectation_bias_at_min": None,
        "searches": searches,
        "provenance": {
            "seed_json": str(args.seed_json),
            "search": "warm-start exact integer-r evaluation",
        },
    }

    def checkpoint():
        payload["searches"] = dedup_searches(searches)
        save_checkpoint(output_path, payload)

    def evaluate(r: int) -> dict:
        r = int(r)
        if r in result_cache:
            item = result_cache[r]
            print(
                f"[expected] seed r={r}: bias={item['expectation_bias']:.6e}, "
                f"threshold={args.epsilon:.6e}",
                flush=True,
            )
            return item
        cache_key = (int(static["n"]), float(args.T), r)
        if cache_key not in base.runner.EVOLUTION_CACHE:
            print(f"[cache] building tilde_V r={r} key={cache_key}", flush=True)
            base.runner.EVOLUTION_CACHE[cache_key] = base.ncc_channel_log.build_tilde_V(static, args.T, r)
        evolution_data = base.runner.EVOLUTION_CACHE[cache_key]
        exact_output_batch, deterministic_output_batch = base.ensure_expected_batches(
            base.ncc_channel_log,
            static,
            evolution_data,
            r,
        )
        expectation_bias, worst_index = base.runner.max_trace_norm_error(
            deterministic_output_batch,
            exact_output_batch,
            threshold=args.epsilon,
        )
        item = {
            "r": r,
            "expectation_bias": float(expectation_bias),
            "worst_index": -1 if worst_index is None else int(worst_index),
            "eta_pair_sum": float(evolution_data["eta_pair_sum"]),
            "pair_scale": float(evolution_data["pair_scale"]),
            "raw_l1_norm_total": float(evolution_data["raw_l1_norm_total"]),
        }
        result_cache[r] = item
        searches.append(item)
        print(
            f"[expected] eval r={r}: bias={item['expectation_bias']:.6e}, "
            f"threshold={args.epsilon:.6e}",
            flush=True,
        )
        checkpoint()
        return item

    failing = sorted([item for item in seed_searches if item["expectation_bias"] > args.epsilon], key=lambda x: x["r"])
    passing = sorted([item for item in seed_searches if item["expectation_bias"] <= args.epsilon], key=lambda x: x["r"])
    if failing and passing:
        low = max(item["r"] for item in failing if item["r"] < min(p["r"] for p in passing))
        high_eval = min(passing, key=lambda x: x["r"])
        high = int(high_eval["r"])
    else:
        low = 0
        high = 1
        high_eval = evaluate(high)
        while high_eval["expectation_bias"] > args.epsilon and high < args.r_max:
            low = high
            high *= 2
            high_eval = evaluate(high)
        if high_eval["expectation_bias"] > args.epsilon:
            raise RuntimeError(f"failed to reach epsilon={args.epsilon} by r={args.r_max}")

    print(f"[search] initial bracket: ({low}, {high}]", flush=True)
    while low + 1 < high:
        mid = (low + high) // 2
        mid_eval = evaluate(mid)
        if mid_eval["expectation_bias"] <= args.epsilon:
            high = mid
            high_eval = mid_eval
        else:
            low = mid
        print(f"[search] bracket now: ({low}, {high}]", flush=True)

    payload["expected_r_min"] = int(high)
    payload["expectation_bias_at_min"] = float(high_eval["expectation_bias"])
    checkpoint()
    print(
        f"[done] expected_r_min={high}, bias={high_eval['expectation_bias']:.6e}, "
        f"saved={output_path}.json",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
