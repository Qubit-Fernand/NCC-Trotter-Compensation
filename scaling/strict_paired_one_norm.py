#!/usr/bin/env python3
"""Evaluate strict paired channel-sampling 1-norms for plotted NCC points.

For K=1 and q0=s0=4, the strict HNCC convention pairs Phi_2, Phi_3,
and Phi_4 together, leaving only (ad_Phi_2)^2 / 2 as a Taylor tail.
This is deliberately different from the current executable channel-log
sampler, which puts Phi_4 in a separate plus/minus tail branch.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
SOURCE_DIR = HERE if (HERE / "NCC_channel_log.py").exists() else HERE / "scaling"
if not SOURCE_DIR.is_dir():
    raise RuntimeError(f"Cannot locate channel sources next to {__file__}")
sys.path.insert(0, str(SOURCE_DIR))

from NCC_channel_log import build_static_data as build_log_static  # noqa: E402
from NCC_channel_original import build_static_data as build_original_static  # noqa: E402


RESULT_SOURCES = (
    ("original", 4, "NCC_channel_original_expected_r_min_*.json"),
    ("log", 4, "NCC_channel_log_expected_r_min_*.json"),
    ("log", 3, "NCC_channel_log_sampling_r_*.json"),
    ("original", 3, "NCC_channel_original_sampling_r_*.json"),
)
EPSILON_TARGETS = (0.1, 0.04, 0.02, 0.01, 0.005, 0.0025, 0.001)
TIME_TARGETS = (0.2, 0.4, 0.6, 0.8, 1.0, 2.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=SOURCE_DIR / "data")
    parser.add_argument("--points-out", type=Path, help="Write the selected plotted points and exit.")
    parser.add_argument("--points", type=Path, help="Selected points JSON, normally emitted by --points-out.")
    parser.add_argument("--original-static-cache", type=Path, required=False)
    parser.add_argument("--log-static-cache", type=Path, required=False)
    parser.add_argument("--n-values", type=int, nargs="+", help="Evaluate only these N values, in this order.")
    parser.add_argument("--output", type=Path, help="Computed strict-pair norm JSON.")
    return parser.parse_args()


def expected_value(payload: dict[str, Any]) -> float | None:
    value = payload.get("expected_r_min")
    if value is not None and float(value) >= 0:
        return float(value)
    values = [float(item) for item in payload.get("expected_r_mins", []) if float(item) >= 0]
    return sum(values) / len(values) if values else None


def load_records(data_dir: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for mode, priority, pattern in RESULT_SOURCES:
        for path in sorted(data_dir.rglob(pattern)):
            if "smoke" in path.parts:
                continue
            payload = json.loads(path.read_text())
            params = payload.get("params", {})
            if not params and all(key in payload for key in ("N", "T", "epsilon")):
                params = {key: payload[key] for key in ("N", "T", "epsilon")}
            if not params:
                continue
            value = expected_value(payload)
            if value is None:
                continue
            records.append(
                {
                    "mode": mode,
                    "priority": priority,
                    "N": int(params["N"]),
                    "T": float(params["T"]),
                    "epsilon": float(params["epsilon"]),
                    "r": value,
                    "source_file": path.name,
                    "repeats": int(params.get("repeats", params.get("requested_repeats", 1))),
                    "trials": int(params.get("trials", 0)),
                }
            )
    return records


def select_plotted_points(data_dir: Path) -> list[dict[str, Any]]:
    records = load_records(data_dir)
    by_key: dict[tuple[str, int, float, float], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_key[(record["mode"], record["N"], record["T"], record["epsilon"])].append(record)

    targets: list[tuple[str, int, float, float, str]] = []
    for mode in ("original", "log"):
        targets.extend((mode, 8, T, 0.001, "T") for T in TIME_TARGETS)
        targets.extend((mode, 8, 2.0, epsilon, "inv_epsilon") for epsilon in EPSILON_TARGETS)
    for mode in ("original", "log"):
        targets.extend((mode, N, 2.0, 0.001, "N") for N in (4, 6))

    selected: dict[tuple[str, int, float, float], dict[str, Any]] = {}
    for mode, n, total_time, epsilon, axis in targets:
        key = (mode, n, total_time, epsilon)
        choices = by_key.get(key, [])
        if not choices:
            raise FileNotFoundError(f"No plotted record for {key}")
        best = max(choices, key=lambda item: (item["priority"], item["repeats"], item["trials"]))
        item = dict(best)
        item["axes"] = [axis]
        if key in selected:
            selected[key]["axes"].append(axis)
        else:
            selected[key] = item
    return sorted(selected.values(), key=lambda item: (item["N"], item["mode"], item["T"], item["epsilon"]))


def load_cache(path: Path | None) -> dict[Any, dict[str, Any]]:
    if path is None:
        return {}
    with gzip.open(path, "rb") as handle:
        raw = pickle.load(handle)
    if not isinstance(raw, dict):
        raise TypeError(f"Static cache {path} is not a dictionary")
    return raw


def pick_static(cache: dict[Any, dict[str, Any]], key: tuple[Any, ...], builder) -> tuple[dict[str, Any], str]:
    if key in cache:
        return cache[key], "static_cache"
    return builder(), "built_from_static_source"


def coefficients_for_n(
    n: int,
    original_cache: dict[Any, dict[str, Any]],
    log_cache: dict[Any, dict[str, Any]],
) -> tuple[dict[str, float], dict[str, str]]:
    original, original_source = pick_static(
        original_cache,
        (n, 1.0, 1.0),
        lambda: build_original_static(n, 1.0, 1.0),
    )
    log, log_source = pick_static(
        log_cache,
        (n, 4, 4, 1.0, 1.0, 1),
        lambda: build_log_static(n, 4, 4, 1.0, 1.0, 1),
    )
    coeffs = {
        "a2_original": float(original["f2_l1"]) / 2.0,
        "a3_original": float(original["f3_l1"]) / 6.0,
        "a2_log": float(log["phi_layer_in_tail_F"][2]["l1_norm"]),
        "a3_log": float(log["phi_layer_in_tail_F"][3]["l1_norm"]),
        "a4_log": float(log["phi_layer_in_tail_F"][4]["l1_norm"]),
    }
    if not math.isclose(coeffs["a2_original"], coeffs["a2_log"], rel_tol=0.0, abs_tol=1e-8):
        raise ValueError(f"Phi_2 mismatch at N={n}: {coeffs}")
    if not math.isclose(coeffs["a3_original"], coeffs["a3_log"], rel_tol=0.0, abs_tol=1e-8):
        raise ValueError(f"Phi_3 mismatch at N={n}: {coeffs}")
    return coeffs, {"original": original_source, "log": log_source}


def evaluate_point(point: dict[str, Any], coeffs: dict[str, float]) -> dict[str, Any]:
    n, total_time, r = int(point["N"]), float(point["T"]), float(point["r"])
    x = total_time / r
    a2, a3, a4 = coeffs["a2_log"], coeffs["a3_log"], coeffs["a4_log"]
    if point["mode"] == "original":
        pair_eta = a2 * x**2 + a3 * x**3
        tail_norm = 0.0
        one_step = 1.0 + 2.0 * pair_eta**2
        formula = "[1 + 2(a2 x^2 + a3 x^3)^2]^r"
    elif point["mode"] == "log":
        pair_eta = a2 * x**2 + a3 * x**3 + a4 * x**4
        tail_norm = 2.0 * a2**2 * x**4
        one_step = 1.0 + 2.0 * pair_eta**2 + tail_norm
        formula = "[1 + 2(a2 x^2 + a3 x^3 + a4 x^4)^2 + 2 a2^2 x^4]^r"
    else:
        raise ValueError(f"Unknown mode {point['mode']}")
    log_total = r * math.log(one_step)
    return {
        **point,
        "x": x,
        "a": {"a2": a2, "a3": a3, "a4": a4},
        "pair_eta": pair_eta,
        "tail_one_step_1norm": tail_norm,
        "one_step_1norm": one_step,
        "log_total_1norm": log_total,
        "total_1norm": math.exp(log_total),
        "formula": formula,
    }


def main() -> None:
    args = parse_args()
    if args.points_out:
        points = select_plotted_points(args.data_dir)
        args.points_out.parent.mkdir(parents=True, exist_ok=True)
        args.points_out.write_text(json.dumps({"points": points}, indent=2) + "\n")
        print(f"wrote {len(points)} plotted points to {args.points_out}")
        return
    if not args.points or not args.output:
        raise SystemExit("Use --points-out, or provide both --points and --output.")

    points = json.loads(args.points.read_text())["points"]
    n_values = args.n_values or sorted({int(point["N"]) for point in points})
    unknown_n = set(n_values) - {int(point["N"]) for point in points}
    if unknown_n:
        raise ValueError(f"No selected points for N={sorted(unknown_n)}")
    original_cache = load_cache(args.original_static_cache)
    log_cache = load_cache(args.log_static_cache)
    all_coefficients: dict[str, Any] = {}
    rows = []
    for n in n_values:
        coeffs, sources = coefficients_for_n(n, original_cache, log_cache)
        all_coefficients[str(n)] = {**coeffs, "sources": sources}
        print(f"N={n}: {all_coefficients[str(n)]}", flush=True)
        rows.extend(evaluate_point(point, coeffs) for point in points if int(point["N"]) == n)

    output = {
        "definition": {
            "K": 1,
            "q0": 4,
            "s0": 4,
            "strict_pairing": "original pairs Phi_2,Phi_3; HNCC pairs Phi_2,Phi_3,Phi_4 and leaves (ad_Phi_2)^2/2 as the Taylor tail",
            "x": "T/r",
        },
        "static_coefficients": all_coefficients,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n")
    print(f"wrote {len(rows)} strict-pair 1-norm rows to {args.output}", flush=True)


if __name__ == "__main__":
    main()
