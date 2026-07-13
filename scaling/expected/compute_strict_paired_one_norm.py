#!/usr/bin/env python3
"""Compute strict-paired channel sampling 1-norms for expected-scan results."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
ONE_NORM_SCRIPT = REPO_ROOT / "1-norm" / "recompute_strict_paired_channel_floors.py"
DEFAULT_DATA_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_JSON = DEFAULT_DATA_DIR / "strict_paired_one_norm_expected_all.json"
DEFAULT_OUTPUT_CSV = DEFAULT_DATA_DIR / "strict_paired_one_norm_expected_all.csv"


def load_one_norm_module():
    spec = importlib.util.spec_from_file_location("strict_paired_floors", ONE_NORM_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {ONE_NORM_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    return parser.parse_args()


def one_step_terms(mode: str, x: float, a2: float, a3: float, a4: float) -> dict[str, float | str]:
    if mode == "original":
        pair_eta = a2 * x**2 + a3 * x**3
        tail_norm = 0.0
        one_step = 1.0 + 2.0 * pair_eta**2
        formula = "[1 + 2(a2 x^2 + a3 x^3)^2]^r"
    elif mode == "log":
        pair_eta = a2 * x**2 + a3 * x**3 + a4 * x**4
        tail_norm = 2.0 * a2**2 * x**4
        one_step = 1.0 + 2.0 * pair_eta**2 + tail_norm
        formula = "[1 + 2(a2 x^2 + a3 x^3 + a4 x^4)^2 + 2 a2^2 x^4]^r"
    else:
        raise ValueError(f"unsupported mode {mode!r}")
    return {
        "pair_eta": pair_eta,
        "direct_tail_one_step_1norm": tail_norm,
        "one_step_1norm": one_step,
        "formula": formula,
    }


def build_row(path: Path, payload: dict[str, Any], coeffs: dict[int, tuple[float, float, float]]) -> dict[str, Any] | None:
    mode = payload.get("mode")
    params = payload.get("params") or {}
    r = payload.get("step_proxy_r_min")
    if mode not in {"original", "log"} or r is None:
        return None

    n = int(params["N"])
    t_total = float(params["T"])
    epsilon = float(params["epsilon"])
    r_int = int(r)
    a2, a3, a4 = coeffs[n]
    x = t_total / r_int
    terms = one_step_terms(mode, x, a2, a3, a4)
    one_step = float(terms["one_step_1norm"])
    log_total = r_int * math.log(one_step)

    return {
        "mode": mode,
        "method": "HNCC" if mode == "log" else "NCC",
        "N": n,
        "T": t_total,
        "epsilon": epsilon,
        "r": r_int,
        "x": x,
        "a2": a2,
        "a3": a3,
        "a4": a4,
        "pair_eta": terms["pair_eta"],
        "direct_tail_one_step_1norm": terms["direct_tail_one_step_1norm"],
        "one_step_1norm": one_step,
        "log_total_1norm": log_total,
        "total_1norm": math.exp(log_total),
        "formula": terms["formula"],
        "canonical_id": payload.get("canonical_id", ""),
        "sweep_id": payload.get("sweep_id", ""),
        "source_host": payload.get("source_host", ""),
        "source_file": path.name,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "mode",
        "method",
        "N",
        "T",
        "epsilon",
        "r",
        "x",
        "a2",
        "a3",
        "a4",
        "pair_eta",
        "direct_tail_one_step_1norm",
        "one_step_1norm",
        "log_total_1norm",
        "total_1norm",
        "duplicate_group_size",
        "duplicate_index",
        "canonical_id",
        "sweep_id",
        "source_host",
        "source_file",
        "formula",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    module = load_one_norm_module()

    input_paths = sorted(args.data_dir.glob("NCC_channel_*_step_proxy_r_min_trace_*.json"))
    payloads = []
    for path in input_paths:
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("mode") in {"original", "log"} and payload.get("step_proxy_r_min") is not None:
            payloads.append((path, payload))

    n_values = sorted({int(payload["params"]["N"]) for _, payload in payloads})
    coeffs = {n: module.phi_norms(n) for n in n_values}
    rows = [row for path, payload in payloads if (row := build_row(path, payload, coeffs)) is not None]
    rows.sort(key=lambda row: (row["mode"], row["N"], row["T"], row["epsilon"], row["r"], row["source_file"]))

    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["mode"], row["N"], row["T"], row["epsilon"])].append(row)
    for group_rows in groups.values():
        group_rows.sort(key=lambda row: row["source_file"])
        for index, row in enumerate(group_rows, start=1):
            row["duplicate_group_size"] = len(group_rows)
            row["duplicate_index"] = index

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "definition": {
            "source": "scaling/expected/NCC_channel_*_step_proxy_r_min_trace_*.json",
            "coefficient_source": str(ONE_NORM_SCRIPT.relative_to(REPO_ROOT)),
            "original_formula": "[1 + 2(a2 x^2 + a3 x^3)^2]^r",
            "hncc_strict_formula": "[1 + 2(a2 x^2 + a3 x^3 + a4 x^4)^2 + 2 a2^2 x^4]^r",
            "x": "T/r where r is step_proxy_r_min from each expected-scan JSON",
        },
        "static_coefficients": {
            str(n): {"a2": a2, "a3": a3, "a4": a4}
            for n, (a2, a3, a4) in coeffs.items()
        },
        "row_count": len(rows),
        "duplicate_parameter_groups": sum(1 for group_rows in groups.values() if len(group_rows) > 1),
        "rows": rows,
    }
    with args.output_json.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
        handle.write("\n")
    write_csv(args.output_csv, rows)

    print(f"wrote {len(rows)} rows to {args.output_json}")
    print(f"wrote {len(rows)} rows to {args.output_csv}")
    print(f"duplicate parameter groups: {output['duplicate_parameter_groups']}")


if __name__ == "__main__":
    main()
