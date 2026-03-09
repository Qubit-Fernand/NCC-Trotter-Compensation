import argparse
import json
import math
from pathlib import Path

import numpy as np

from ncc_scaling_experiment import build_system_cache, find_min_segments, fit_power_law


def parse_args():
    parser = argparse.ArgumentParser(description="Reproduce plot_original_scaling.ipynb data generation.")
    parser.add_argument("--out-dir", type=Path, default=Path("data"), help="output directory")
    parser.add_argument("--tag", type=str, default="", help="optional suffix in output filenames")
    parser.add_argument("--r-max", type=int, default=512, help="maximal r allowed during binary search")
    parser.add_argument("--J", type=float, default=1.0, help="interaction strength")
    parser.add_argument("--h", type=float, default=1.0, help="field strength")
    return parser.parse_args()


def gate_proxy(n, r_min, kappa=1.0):
    return r_min * (2.0 * kappa * n + 4.0)


def save_results(out_base: Path, payload: dict):
    json_path = Path(f"{out_base}.json")
    npz_path = Path(f"{out_base}.npz")
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    np.savez(
        npz_path,
        n_values=np.array(payload["n_sweep"]["values"], dtype=float),
        n_r_min=np.array(payload["n_sweep"]["r_min"], dtype=int),
        n_err=np.array(payload["n_sweep"]["err"], dtype=float),
        n_gate=np.array(payload["n_sweep"]["gate"], dtype=float),
        t_values=np.array(payload["t_sweep"]["values"], dtype=float),
        t_r_min=np.array(payload["t_sweep"]["r_min"], dtype=int),
        t_err=np.array(payload["t_sweep"]["err"], dtype=float),
        t_gate=np.array(payload["t_sweep"]["gate"], dtype=float),
        eps_values=np.array(payload["eps_sweep"]["values"], dtype=float),
        eps_r_min=np.array(payload["eps_sweep"]["r_min"], dtype=int),
        eps_err=np.array(payload["eps_sweep"]["err"], dtype=float),
        eps_gate=np.array(payload["eps_sweep"]["gate"], dtype=float),
        slope_n=float(payload["fits"]["slope_n"]),
        slope_t=float(payload["fits"]["slope_t"]),
        slope_eps=float(payload["fits"]["slope_eps"]),
    )
    return json_path, npz_path


def main():
    args = parse_args()

    n_values = [4, 5, 6, 7, 8, 9, 10]
    t_values = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    eps_values = [2e-2, 1.5e-2, 1e-2, 7e-3, 5e-3, 3e-3]

    fixed_t_for_n = 1.0
    fixed_eps_for_n = 1e-2
    fixed_n_for_t = 6
    fixed_eps_for_t = 1e-2
    fixed_n_for_eps = 6
    fixed_t_for_eps = 1.0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    out_base = args.out_dir / f"plot_original_scaling_data{suffix}"

    caches = {
        n: build_system_cache(n, j=args.J, h=args.h)
        for n in sorted(set(n_values + [fixed_n_for_t, fixed_n_for_eps]))
    }

    payload = {
        "script": "plot_original_scaling_data.py",
        "params": {
            "J": args.J,
            "h": args.h,
            "r_max": args.r_max,
        },
        "n_sweep": {"values": n_values, "r_min": [], "err": [], "gate": []},
        "t_sweep": {"values": t_values, "r_min": [], "err": [], "gate": []},
        "eps_sweep": {"values": eps_values, "r_min": [], "err": [], "gate": []},
        "fits": {},
    }

    for n in n_values:
        r_min, err = find_min_segments(caches[n], fixed_t_for_n, fixed_eps_for_n, r_max=args.r_max)
        gate = gate_proxy(n, r_min)
        payload["n_sweep"]["r_min"].append(int(r_min))
        payload["n_sweep"]["err"].append(float(err))
        payload["n_sweep"]["gate"].append(float(gate))
        print(f"N sweep: N={n}, r_min={r_min}, err={err:.3e}, G_proxy={gate:.1f}")

    for t_total in t_values:
        r_min, err = find_min_segments(caches[fixed_n_for_t], t_total, fixed_eps_for_t, r_max=args.r_max)
        gate = gate_proxy(fixed_n_for_t, r_min)
        payload["t_sweep"]["r_min"].append(int(r_min))
        payload["t_sweep"]["err"].append(float(err))
        payload["t_sweep"]["gate"].append(float(gate))
        print(f"T sweep: T={t_total}, r_min={r_min}, err={err:.3e}, G_proxy={gate:.1f}")

    for eps in eps_values:
        r_min, err = find_min_segments(caches[fixed_n_for_eps], fixed_t_for_eps, eps, r_max=args.r_max)
        gate = gate_proxy(fixed_n_for_eps, r_min)
        payload["eps_sweep"]["r_min"].append(int(r_min))
        payload["eps_sweep"]["err"].append(float(err))
        payload["eps_sweep"]["gate"].append(float(gate))
        print(f"eps sweep: eps={eps}, r_min={r_min}, err={err:.3e}, G_proxy={gate:.1f}")

    slope_n, prefactor_n = fit_power_law(
        np.array(payload["n_sweep"]["values"], dtype=float),
        np.array(payload["n_sweep"]["gate"], dtype=float),
    )
    slope_t, prefactor_t = fit_power_law(
        np.array(payload["t_sweep"]["values"], dtype=float),
        np.array(payload["t_sweep"]["gate"], dtype=float),
    )
    slope_eps, prefactor_eps = fit_power_law(
        np.array(payload["eps_sweep"]["values"], dtype=float),
        np.array(payload["eps_sweep"]["gate"], dtype=float),
    )
    payload["fits"] = {
        "slope_n": float(slope_n),
        "prefactor_n": float(prefactor_n),
        "slope_t": float(slope_t),
        "prefactor_t": float(prefactor_t),
        "slope_eps": float(slope_eps),
        "prefactor_eps": float(prefactor_eps),
        "theory_n": 5.0 / 3.0,
        "theory_t": 4.0 / 3.0,
        "theory_eps": -1.0 / 3.0,
    }

    json_path, npz_path = save_results(out_base, payload)
    print(f"fitted slope (N): {slope_n:.3f}")
    print(f"fitted slope (T): {slope_t:.3f}")
    print(f"fitted slope (eps): {slope_eps:.3f}")
    print(f"saved json to: {json_path}")
    print(f"saved npz to: {npz_path}")


if __name__ == "__main__":
    main()
