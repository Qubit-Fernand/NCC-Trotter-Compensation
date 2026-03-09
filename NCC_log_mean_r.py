import argparse
import json
import math
from pathlib import Path

import numpy as np

from NCC_log import find_min_segments_log
from ncc_scaling_experiment import fit_power_law


def parse_args():
    parser = argparse.ArgumentParser(description="Reproduce plot_log_scaling.ipynb data generation.")
    parser.add_argument("--out-dir", type=Path, default=Path("data"), help="output directory")
    parser.add_argument("--tag", type=str, default="", help="optional suffix in output filenames")
    parser.add_argument("--r-max", type=int, default=512, help="maximal r allowed during binary search")
    parser.add_argument("--J", type=float, default=1.0, help="interaction strength")
    parser.add_argument("--h", type=float, default=1.0, help="field strength")
    parser.add_argument("--sampling", choices=["uniform", "weighted"], default="weighted")
    parser.add_argument("--s0", type=int, default=0, help="override truncation order")
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
        eps_log_inv=np.array(payload["eps_sweep"]["log_inv_eps"], dtype=float),
        eps_r_min=np.array(payload["eps_sweep"]["r_min"], dtype=int),
        eps_err=np.array(payload["eps_sweep"]["err"], dtype=float),
        eps_gate=np.array(payload["eps_sweep"]["gate"], dtype=float),
        slope_n=float(payload["fits"]["slope_n"]),
        slope_t=float(payload["fits"]["slope_t"]),
        slope_log_eps=float(payload["fits"]["slope_log_eps"]),
    )
    return json_path, npz_path


def main():
    args = parse_args()

    n_values = [4, 5, 6]
    t_values = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    eps_values = [5e-2, 4e-2, 3e-2, 2.5e-2, 2e-2, 1.5e-2, 1e-2]

    fixed_t_for_n = 1.0
    fixed_eps_for_n = 1e-2
    fixed_n_for_t = 4
    fixed_eps_for_t = 1e-2
    fixed_n_for_eps = 4
    fixed_t_for_eps = 1.0

    args.out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    s0_label = args.s0 if args.s0 > 0 else "auto"
    out_base = args.out_dir / f"plot_log_scaling_data_{args.sampling}_s0{s0_label}{suffix}"

    payload = {
        "script": "plot_log_scaling_data.py",
        "params": {
            "J": args.J,
            "h": args.h,
            "r_max": args.r_max,
            "sampling": args.sampling,
            "s0": args.s0,
        },
        "n_sweep": {"values": n_values, "r_min": [], "err": [], "gate": []},
        "t_sweep": {"values": t_values, "r_min": [], "err": [], "gate": []},
        "eps_sweep": {"values": eps_values, "log_inv_eps": [], "r_min": [], "err": [], "gate": []},
        "fits": {},
    }

    for n in n_values:
        r_min, err = find_min_segments_log(
            n,
            fixed_t_for_n,
            fixed_eps_for_n,
            j=args.J,
            h=args.h,
            sampling=args.sampling,
            r_max=args.r_max,
            s0=args.s0 or None,
        )
        gate = gate_proxy(n, r_min)
        payload["n_sweep"]["r_min"].append(int(r_min))
        payload["n_sweep"]["err"].append(float(err))
        payload["n_sweep"]["gate"].append(float(gate))
        print(f"N sweep: N={n}, r_min={r_min}, err={err:.3e}, G_proxy={gate:.1f}")

    for t_total in t_values:
        r_min, err = find_min_segments_log(
            fixed_n_for_t,
            t_total,
            fixed_eps_for_t,
            j=args.J,
            h=args.h,
            sampling=args.sampling,
            r_max=args.r_max,
            s0=args.s0 or None,
        )
        gate = gate_proxy(fixed_n_for_t, r_min)
        payload["t_sweep"]["r_min"].append(int(r_min))
        payload["t_sweep"]["err"].append(float(err))
        payload["t_sweep"]["gate"].append(float(gate))
        print(f"T sweep: T={t_total}, r_min={r_min}, err={err:.3e}, G_proxy={gate:.1f}")

    for eps in eps_values:
        log_inv_eps = math.log(1.0 / eps)
        r_min, err = find_min_segments_log(
            fixed_n_for_eps,
            fixed_t_for_eps,
            eps,
            j=args.J,
            h=args.h,
            sampling=args.sampling,
            r_max=args.r_max,
            s0=args.s0 or None,
        )
        gate = gate_proxy(fixed_n_for_eps, r_min)
        payload["eps_sweep"]["log_inv_eps"].append(float(log_inv_eps))
        payload["eps_sweep"]["r_min"].append(int(r_min))
        payload["eps_sweep"]["err"].append(float(err))
        payload["eps_sweep"]["gate"].append(float(gate))
        print(
            f"eps sweep: eps={eps}, log(1/eps)={log_inv_eps:.3f}, "
            f"r_min={r_min}, err={err:.3e}, G_proxy={gate:.1f}"
        )

    slope_n, prefactor_n = fit_power_law(
        np.array(payload["n_sweep"]["values"], dtype=float),
        np.array(payload["n_sweep"]["gate"], dtype=float),
    )
    slope_t, prefactor_t = fit_power_law(
        np.array(payload["t_sweep"]["values"], dtype=float),
        np.array(payload["t_sweep"]["gate"], dtype=float),
    )
    slope_log_eps, prefactor_log_eps = fit_power_law(
        np.array(payload["eps_sweep"]["log_inv_eps"], dtype=float),
        np.array(payload["eps_sweep"]["gate"], dtype=float),
    )
    payload["fits"] = {
        "slope_n": float(slope_n),
        "prefactor_n": float(prefactor_n),
        "slope_t": float(slope_t),
        "prefactor_t": float(prefactor_t),
        "slope_log_eps": float(slope_log_eps),
        "prefactor_log_eps": float(prefactor_log_eps),
        "theory_n": 5.0 / 3.0,
        "theory_t": 4.0 / 3.0,
        "theory_log_eps": 1.0,
    }

    json_path, npz_path = save_results(out_base, payload)
    print(f"fitted slope (N): {slope_n:.3f}")
    print(f"fitted slope (T): {slope_t:.3f}")
    print(f"fitted slope vs log(1/eps): {slope_log_eps:.3f}")
    print(f"saved json to: {json_path}")
    print(f"saved npz to: {npz_path}")


if __name__ == "__main__":
    main()
