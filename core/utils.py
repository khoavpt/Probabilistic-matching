from __future__ import annotations

import json
from pathlib import Path
import numpy as np


PARAMETER_NAMES = ("a1", "a2", "b1", "b2", "k_s", "k_m", "t1", "t2", "t_m")


def as_parameter_array(params):
    if isinstance(params, dict):
        if "params" in params:
            params = params["params"]
        else:
            params = [params[name] for name in PARAMETER_NAMES]
    params = np.asarray(params, dtype=np.float64).reshape(-1)
    if len(params) == 11:
        params = params[:9]
    return params


def load_calibration(path, matcher=None):
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if matcher is not None:
        payload = payload["matchers"][matcher]

    return as_parameter_array(payload)


def confidence_risk(mconf):
    return -np.log(np.clip(np.asarray(mconf, dtype=np.float64), 1e-8, 1.0))


def fine_posterior_weights_from_params(params, std_c, std_f, mconf, err, H, W):
    a1, a2, b1, b2, k_s, k_m, t1, t2, t_m = as_parameter_array(params)
    std_c = np.asarray(std_c, dtype=np.float64)
    std_f = np.asarray(std_f, dtype=np.float64)
    mconf = np.asarray(mconf, dtype=np.float64).reshape(-1)
    err = np.asarray(err, dtype=np.float64)
    conf_term = k_m * (confidence_risk(mconf) - t_m)

    std_c_norm_x = std_c[:, 0] / np.float64(W) * 100.0
    std_c_norm_y = std_c[:, 1] / np.float64(H) * 100.0

    alpha_x = 1.0 / (1.0 + np.exp(-(k_s * (std_c_norm_x - t1) + conf_term)))
    alpha_y = 1.0 / (1.0 + np.exp(-(k_s * (std_c_norm_y - t2) + conf_term)))

    scale_fine_x = np.maximum(np.sqrt(b1) * std_f[:, 0], 1e-6)
    scale_fine_y = np.maximum(np.sqrt(b2) * std_f[:, 1], 1e-6)
    scale_coarse_x = np.maximum(np.sqrt(a1) * std_c[:, 0], 1e-6)
    scale_coarse_y = np.maximum(np.sqrt(a2) * std_c[:, 1], 1e-6)

    log_fine_joint = (
        np.log(1.0 - alpha_x + 1e-8)
        - np.abs(err[:, 0]) / scale_fine_x
        - np.log(2.0 * scale_fine_x)
    )
    log_coarse_joint = (
        np.log(alpha_x + 1e-8)
        - np.abs(err[:, 0]) / scale_coarse_x
        - np.log(2.0 * scale_coarse_x)
    )

    log_fine_joint += (
        np.log(1.0 - alpha_y + 1e-8)
        - np.abs(err[:, 1]) / scale_fine_y
        - np.log(2.0 * scale_fine_y)
    )
    log_coarse_joint += (
        np.log(alpha_y + 1e-8)
        - np.abs(err[:, 1]) / scale_coarse_y
        - np.log(2.0 * scale_coarse_y)
    )

    return np.exp(log_fine_joint - np.logaddexp(log_fine_joint, log_coarse_joint))
