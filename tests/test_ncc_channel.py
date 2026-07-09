import sys
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scaling.NCC_channel_log import build_static_data, build_tilde_V, paired_channel_parameters


def test_paired_channel_angle_uses_tangent_parameterization():
    eta = 0.37
    params = paired_channel_parameters(eta)

    assert math.isclose(math.tan(params["theta"]), eta, rel_tol=1e-14, abs_tol=1e-14)


def test_ncc_channel_action_on_rho_smoke():
    static = build_static_data(
        n=2,
        q0=4,
        s0=4,
        j=1.0,
        h=1.0,
        K=1,
    )
    evolution_data = build_tilde_V(static, t_total=0.2, r=4)

    assert static["pair_validation_errors"][2] < 1e-12
    assert static["pair_validation_errors"][3] < 1e-12
    assert evolution_data["validation_error"] < 1e-12
    assert evolution_data["deterministic_bias"] < evolution_data["uncompensated_total_error"]
