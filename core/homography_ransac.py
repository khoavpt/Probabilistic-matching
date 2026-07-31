import cv2
import numpy as np

from .utils import fine_posterior_weights_from_params, confidence_risk

def normalize_points(pts):
    mean = pts.mean(axis=0)
    std = pts.std(axis=0).mean() + 1e-8

    s = np.sqrt(2.0) / std
    T = np.array(
        [
            [s, 0.0, -s * mean[0]],
            [0.0, s, -s * mean[1]],
            [0.0, 0.0, 1.0],
        ]
    )

    pts_h = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
    pts_n = (T @ pts_h.T).T

    return pts_n[:, :2], T


def compute_homography(pts0, pts1):
    pts0_n, T0 = normalize_points(np.asarray(pts0, dtype=np.float64))
    pts1_n, T1 = normalize_points(np.asarray(pts1, dtype=np.float64))

    x = pts0_n[:, 0]
    y = pts0_n[:, 1]
    u = pts1_n[:, 0]
    v = pts1_n[:, 1]
    ones = np.ones_like(x)
    zeros = np.zeros_like(x)

    A = np.empty((2 * len(pts0_n), 9), dtype=np.float64)
    A[0::2] = np.stack(
        [-x, -y, -ones, zeros, zeros, zeros, x * u, y * u, u], axis=1
    )
    A[1::2] = np.stack(
        [zeros, zeros, zeros, -x, -y, -ones, x * v, y * v, v], axis=1
    )
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)

    H = np.linalg.inv(T1) @ H @ T0
    return H / (H[2, 2] + 1e-8)


def compute_homography_weighted(pts0, pts1, weights):
    pts0 = np.asarray(pts0, dtype=np.float64)
    pts1 = np.asarray(pts1, dtype=np.float64)
    pts0_n, T0 = normalize_points(pts0)
    pts1_n, T1 = normalize_points(pts1)

    weights = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(weights) & (weights > 0)
    if valid.sum() < 4:
        return compute_homography(pts0, pts1)

    pts0_n = pts0_n[valid]
    pts1_n = pts1_n[valid]
    weights = weights[valid]

    x = pts0_n[:, 0]
    y = pts0_n[:, 1]
    u = pts1_n[:, 0]
    v = pts1_n[:, 1]
    ones = np.ones_like(x)
    zeros = np.zeros_like(x)

    A0 = np.stack([-x, -y, -ones, zeros, zeros, zeros, x * u, y * u, u], axis=1)
    A1 = np.stack([zeros, zeros, zeros, -x, -y, -ones, x * v, y * v, v], axis=1)

    normal = A0.T @ (A0 * weights[:, None]) + A1.T @ (A1 * weights[:, None])
    _, eigvecs = np.linalg.eigh(normal)
    H = eigvecs[:, 0].reshape(3, 3)

    H = np.linalg.inv(T1) @ H @ T0
    return H / (H[2, 2] + 1e-8)


def reprojection_error(H, pts0, pts1):
    H_inv = np.linalg.inv(np.asarray(H, dtype=np.float64))
    pts1_h = np.concatenate([pts1, np.ones((len(pts1), 1))], axis=1)
    proj = (H_inv @ pts1_h.T).T
    proj = proj[:, :2] / (proj[:, 2:] + 1e-8)
    return np.asarray(pts0, dtype=np.float64) - proj


def reprojection_error_sqr(H, pts0, pts1):
    pts0_h = np.concatenate([pts0, np.ones((len(pts0), 1))], axis=1)
    proj = (np.asarray(H, dtype=np.float64) @ pts0_h.T).T
    proj = proj[:, :2] / (proj[:, 2:] + 1e-8)
    err = np.asarray(pts1, dtype=np.float64) - proj
    return np.sum(err**2, axis=1)


def gate_alpha(std_c_norm_d, mconf, k_s, k_m, t_s, t_m):
    z = k_s * (std_c_norm_d - t_s) + k_m * (confidence_risk(mconf) - t_m)
    z = np.clip(z, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def normalize_coarse_std(std_c, W, H):
    image_size = np.array([W, H], dtype=np.float64)
    return np.asarray(std_c, dtype=np.float64) / image_size * 100.0


def mixture_log_likelihood_from_params(params, std_c, std_f, mconf, err):
    a1, a2, b1, b2, k_s, k_m, t1, t2, t_m, W, H = np.asarray(
        params, dtype=np.float64
    )
    a = [a1, a2]
    b = [b1, b2]
    t = [t1, t2]

    log_likelihood = 0.0
    std_c_norm = normalize_coarse_std(std_c, W, H)

    for d in range(2):
        alpha = gate_alpha(std_c_norm[:, d], mconf, k_s, k_m, t[d], t_m)
        scale_fine = np.maximum(np.sqrt(b[d]) * std_f[:, d], 1e-6)
        scale_coarse = np.maximum(np.sqrt(a[d]) * std_c[:, d], 1e-6)

        log_p_fine = -np.abs(err[:, d]) / scale_fine - np.log(2.0 * scale_fine)
        log_p_coarse = -np.abs(err[:, d]) / scale_coarse - np.log(2.0 * scale_coarse)

        log_mix = np.logaddexp(
            np.log(1.0 - alpha + 1e-8) + log_p_fine,
            np.log(alpha + 1e-8) + log_p_coarse,
        )
        log_likelihood += log_mix
    return log_likelihood




def compute_calibrated_std(
    std_c, std_f, mconf, a1, a2, b1, b2, k_s, k_m, t1, t2, t_m, W, H
):
    std_xy = []
    a = [a1, a2]
    b = [b1, b2]
    t = [t1, t2]
    std_c_norm = normalize_coarse_std(std_c, W, H)

    for d in range(2):
        alpha = gate_alpha(std_c_norm[:, d], mconf, k_s, k_m, t[d], t_m)
        scale_fine = np.maximum(np.sqrt(b[d]) * std_f[:, d], 1e-6)
        scale_coarse = np.maximum(np.sqrt(a[d]) * std_c[:, d], 1e-6)
        std_xy.append((1.0 - alpha) * scale_fine + alpha * scale_coarse)

    return np.stack(std_xy, axis=1)


def _opencv_method(method):
    aliases = {
        "ransac": "RANSAC",
        "lo-ransac": "USAC_DEFAULT",
        "prosac": "USAC_PROSAC",
        "magsac": "USAC_MAGSAC",
        "magsac++": "USAC_MAGSAC",
        "usac": "USAC_DEFAULT",
        "usac-fast": "USAC_FAST",
        "usac-accurate": "USAC_ACCURATE",
    }
    if isinstance(method, str):
        attribute = aliases.get(method.lower(), method.upper())
        if not hasattr(cv2, attribute):
            raise ValueError(
                f"Estimator '{method}' requires OpenCV with cv2.{attribute}."
            )
        return int(getattr(cv2, attribute))
    return int(method)


def find_homography_opencv(
    pts0, pts1, threshold=3.0, max_iters=2000, confidence=0.999, method="ransac"
):
    H, mask = cv2.findHomography(
        np.asarray(pts0, dtype=np.float32),
        np.asarray(pts1, dtype=np.float32),
        _opencv_method(method),
        ransacReprojThreshold=threshold,
        maxIters=max_iters,
        confidence=confidence,
    )
    if H is None or mask is None:
        return None, np.zeros(len(pts0), dtype=bool)
    return H / (H[2, 2] + 1e-8), mask.ravel().astype(bool)


def find_homography_ransac(
    pts0, pts1, threshold=3.0, max_iters=2000, confidence=0.999, method="ransac"
):
    return find_homography_opencv(
        pts0,
        pts1,
        threshold=threshold,
        max_iters=max_iters,
        confidence=confidence,
        method=method,
    )


def find_homography_python_ransac(
    pts0, pts1, threshold=3.0, max_iters=2000, confidence=0.999
):
    N = len(pts0)
    if N < 4:
        return None, np.zeros(N, dtype=bool)

    best_H = None
    best_inliers = None
    best_inlier_count = 0
    thr_sqr = threshold * threshold

    for _ in range(max_iters):
        idx = np.random.choice(N, 4, replace=False)
        try:
            H = compute_homography(pts0[idx], pts1[idx])
        except np.linalg.LinAlgError:
            continue

        err_sqr = reprojection_error_sqr(H, pts0, pts1)
        inliers = err_sqr <= thr_sqr
        count = inliers.sum()

        if count > best_inlier_count:
            best_inlier_count = count
            best_H = H
            best_inliers = inliers

    if best_H is None or best_inlier_count < 4:
        return None, np.zeros(N, dtype=bool)

    H = compute_homography(pts0[best_inliers], pts1[best_inliers])
    err_sqr = reprojection_error_sqr(H, pts0, pts1)
    final_inliers = err_sqr <= thr_sqr
    return H, final_inliers


def find_homography_prob_ransac(
    pts0,
    pts1,
    std_c,
    std_f,
    mconf,
    params,
    image_size,
    threshold=3.0,
    max_iters=1000,
    confidence=0.999,
    method="ransac",
):
    H, inliers = find_homography_opencv(
        pts0,
        pts1,
        threshold=threshold,
        max_iters=max_iters,
        confidence=confidence,
        method=method,
    )
    if H is None or inliers.sum() < 4:
        return H, inliers

    W, H_img = image_size
    err = reprojection_error(H, pts0, pts1)
    weights = fine_posterior_weights_from_params(
        params, std_c, std_f, mconf, err, H_img, W
    )
    H = compute_homography_weighted(pts0, pts1, weights)
    err_sqr = reprojection_error_sqr(H, pts0, pts1)
    final_inliers = err_sqr <= threshold * threshold
    return H, final_inliers


def project_points(H, pts):
    pts = np.asarray(pts, dtype=np.float64)
    pts_h = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)
    proj = (np.asarray(H, dtype=np.float64) @ pts_h.T).T
    return proj[:, :2] / (proj[:, 2:] + 1e-8)


def estimate_homography(
    pts0,
    pts1,
    estimator="ransac",
    threshold=2.0,
    max_iterations=10_000,
    confidence=0.999,
):
    return find_homography_opencv(
        np.asarray(pts0, dtype=np.float64),
        np.asarray(pts1, dtype=np.float64),
        threshold=float(threshold),
        max_iters=int(max_iterations),
        confidence=float(confidence),
        method=estimator,
    )


def refit_homography_core(
    H,
    pts0,
    pts1,
    std_c,
    std_f,
    mconf,
    params,
    image_size,
    threshold=2.0,
):
    params = np.asarray(params, dtype=np.float64).reshape(-1)
    if len(params) == 11:
        params = params[:9]
    if len(params) != 9:
        raise ValueError(f"Expected 9 calibration values, got {len(params)}.")

    pts0 = np.asarray(pts0, dtype=np.float64)
    pts1 = np.asarray(pts1, dtype=np.float64)
    W, H_img = image_size
    err = reprojection_error(H, pts0, pts1)
    weights = fine_posterior_weights_from_params(
        params,
        std_c,
        std_f,
        mconf,
        err,
        H_img,
        W,
    )

    H_refit = compute_homography_weighted(pts0, pts1, weights)
    if H_refit is None or not np.isfinite(H_refit).all():
        return None, np.zeros(len(pts0), dtype=bool), weights

    final_inliers = reprojection_error_sqr(H_refit, pts0, pts1) <= threshold * threshold
    return H_refit, final_inliers, weights


def estimate_homography_core(
    pts0,
    pts1,
    std_c,
    std_f,
    mconf,
    params,
    image_size,
    estimator="ransac",
    threshold=2.0,
    max_iterations=10_000,
    confidence=0.999,
):
    H, inliers = estimate_homography(
        pts0,
        pts1,
        estimator=estimator,
        threshold=threshold,
        max_iterations=max_iterations,
        confidence=confidence,
    )
    if H is None:
        return None, inliers, np.zeros(len(pts0), dtype=np.float64)

    return refit_homography_core(
        H,
        pts0,
        pts1,
        std_c,
        std_f,
        mconf,
        params,
        image_size,
        threshold=threshold,
    )
