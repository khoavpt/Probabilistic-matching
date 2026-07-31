import cv2
import numpy as np

from .utils import fine_posterior_weights_from_params


def rodrigues_project(points3d, rvec, tvec, K, dist=None):
    dist = np.zeros(4, dtype=np.float64) if dist is None else dist
    proj, _ = cv2.projectPoints(
        np.asarray(points3d, dtype=np.float64),
        np.asarray(rvec, dtype=np.float64),
        np.asarray(tvec, dtype=np.float64),
        np.asarray(K, dtype=np.float64),
        np.asarray(dist, dtype=np.float64),
    )
    return proj.reshape(-1, 2)


def reprojection_error_pnp(points2d, points3d, rvec, tvec, K, dist=None):
    return np.asarray(points2d, dtype=np.float64) - rodrigues_project(
        points3d, rvec, tvec, K, dist
    )


def find_pnp_ransac(
    points2d,
    points3d,
    K,
    dist=None,
    threshold=20.0,
    max_iters=1000,
    confidence=0.999,
):
    if len(points2d) < 6:
        return None, None, np.zeros(len(points2d), dtype=bool)

    dist = np.zeros(4, dtype=np.float64) if dist is None else dist
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        np.asarray(points3d, dtype=np.float64),
        np.asarray(points2d, dtype=np.float64),
        np.asarray(K, dtype=np.float64),
        np.asarray(dist, dtype=np.float64),
        iterationsCount=max_iters,
        reprojectionError=threshold,
        confidence=confidence,
        flags=cv2.SOLVEPNP_EPNP,
    )
    if not ok or inliers is None:
        return None, None, np.zeros(len(points2d), dtype=bool)

    inlier_mask = np.zeros(len(points2d), dtype=bool)
    inlier_mask[inliers.ravel()] = True
    if inlier_mask.sum() >= 6:
        try:
            ok, rvec, tvec = cv2.solvePnP(
                np.asarray(points3d[inlier_mask], dtype=np.float64),
                np.asarray(points2d[inlier_mask], dtype=np.float64),
                np.asarray(K, dtype=np.float64),
                np.asarray(dist, dtype=np.float64),
                rvec,
                tvec,
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
        except cv2.error:
            ok = False
        if not ok:
            rvec, tvec = rvec.reshape(3, 1), tvec.reshape(3, 1)

    err = reprojection_error_pnp(points2d, points3d, rvec, tvec, K, dist)
    final_inliers = np.sum(err * err, axis=1) <= threshold * threshold
    return rvec.reshape(3, 1), tvec.reshape(3, 1), final_inliers


def weighted_refine_pnp(points2d, points3d, weights, K, dist, rvec, tvec):
    valid = np.isfinite(weights) & (weights > 0)
    if valid.sum() < 6:
        return rvec, tvec

    points2d = np.asarray(points2d[valid], dtype=np.float64)
    points3d = np.asarray(points3d[valid], dtype=np.float64)
    weights = np.asarray(weights[valid], dtype=np.float64)
    sqrt_w = np.sqrt(weights / np.maximum(np.mean(weights), 1e-12))
    x0 = np.concatenate([np.asarray(rvec).reshape(3), np.asarray(tvec).reshape(3)])

    def residual(x):
        rv = x[:3].reshape(3, 1)
        tv = x[3:].reshape(3, 1)
        err = reprojection_error_pnp(points2d, points3d, rv, tv, K, dist)
        return (err * sqrt_w[:, None]).reshape(-1)

    try:
        from scipy.optimize import least_squares

        result = least_squares(
            residual,
            x0,
            method="trf",
            loss="linear",
            max_nfev=50,
            xtol=1e-8,
            ftol=1e-8,
            gtol=1e-8,
        )
        if result.success:
            return result.x[:3].reshape(3, 1), result.x[3:].reshape(3, 1)
    except Exception:
        pass

    x = x0.copy()
    damping = 1e-3
    eps = 1e-6
    last_cost = np.inf
    for _ in range(20):
        r = residual(x)
        cost = float(r @ r)
        if not np.isfinite(cost):
            break

        J = np.empty((len(r), 6), dtype=np.float64)
        for j in range(6):
            step = eps * max(1.0, abs(x[j]))
            xp = x.copy()
            xm = x.copy()
            xp[j] += step
            xm[j] -= step
            J[:, j] = (residual(xp) - residual(xm)) / (2.0 * step)

        A = J.T @ J + damping * np.eye(6)
        b = -J.T @ r
        try:
            delta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            break

        candidate = x + delta
        candidate_cost = float(residual(candidate) @ residual(candidate))
        if candidate_cost < cost:
            x = candidate
            damping *= 0.5
            if abs(last_cost - candidate_cost) < 1e-6:
                break
            last_cost = candidate_cost
        else:
            damping *= 5.0

    return x[:3].reshape(3, 1), x[3:].reshape(3, 1)


def find_pnp_prob_ransac(
    points2d,
    points3d,
    std_c,
    std_f,
    mconf,
    params,
    image_size,
    K,
    dist=None,
    threshold=20.0,
    max_iters=1000,
    confidence=0.999,
):
    rvec, tvec, inliers = find_pnp_ransac(
        points2d,
        points3d,
        K,
        dist=dist,
        threshold=threshold,
        max_iters=max_iters,
        confidence=confidence,
    )
    if rvec is None or inliers.sum() < 6:
        return rvec, tvec, inliers

    W, H = image_size
    err = reprojection_error_pnp(points2d, points3d, rvec, tvec, K, dist)
    weights = fine_posterior_weights_from_params(
        params, std_c, std_f, mconf, err, H, W
    )

    rvec, tvec = weighted_refine_pnp(points2d, points3d, weights, K, dist, rvec, tvec)
    err = reprojection_error_pnp(points2d, points3d, rvec, tvec, K, dist)
    final_inliers = np.sum(err * err, axis=1) <= threshold * threshold
    return rvec, tvec, final_inliers


def rotation_to_qvec(R):
    qvec = np.empty(4, dtype=np.float64)
    trace = np.trace(R)
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2.0
        qvec[0] = 0.25 * s
        qvec[1] = (R[2, 1] - R[1, 2]) / s
        qvec[2] = (R[0, 2] - R[2, 0]) / s
        qvec[3] = (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            qvec[0] = (R[2, 1] - R[1, 2]) / s
            qvec[1] = 0.25 * s
            qvec[2] = (R[0, 1] + R[1, 0]) / s
            qvec[3] = (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            qvec[0] = (R[0, 2] - R[2, 0]) / s
            qvec[1] = (R[0, 1] + R[1, 0]) / s
            qvec[2] = 0.25 * s
            qvec[3] = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            qvec[0] = (R[1, 0] - R[0, 1]) / s
            qvec[1] = (R[0, 2] + R[2, 0]) / s
            qvec[2] = (R[1, 2] + R[2, 1]) / s
            qvec[3] = 0.25 * s
    if qvec[0] < 0:
        qvec *= -1
    return qvec / np.linalg.norm(qvec)


def pnp_pose_to_qt(rvec, tvec):
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64))
    return rotation_to_qvec(R), np.asarray(tvec, dtype=np.float64).reshape(3)


def project_points(points3d, rvec, tvec, K, dist=None):
    return rodrigues_project(
        np.asarray(points3d, dtype=np.float64),
        np.asarray(rvec, dtype=np.float64).reshape(3, 1),
        np.asarray(tvec, dtype=np.float64).reshape(3, 1),
        np.asarray(K, dtype=np.float64).reshape(3, 3),
        None if dist is None else np.asarray(dist, dtype=np.float64),
    )


def pnp_residual(points3d, points2d, rvec, tvec, K, dist=None):
    return reprojection_error_pnp(
        np.asarray(points2d, dtype=np.float64),
        np.asarray(points3d, dtype=np.float64),
        np.asarray(rvec, dtype=np.float64).reshape(3, 1),
        np.asarray(tvec, dtype=np.float64).reshape(3, 1),
        np.asarray(K, dtype=np.float64).reshape(3, 3),
        None if dist is None else np.asarray(dist, dtype=np.float64),
    )


def estimate_pnp(
    points3d,
    points2d,
    K,
    dist=None,
    threshold=12.0,
    confidence=0.999,
    max_iterations=10_000,
):
    rvec, tvec, inliers = find_pnp_ransac(
        np.asarray(points2d, dtype=np.float64),
        np.asarray(points3d, dtype=np.float64),
        np.asarray(K, dtype=np.float64).reshape(3, 3),
        dist=None if dist is None else np.asarray(dist, dtype=np.float64),
        threshold=float(threshold),
        max_iters=int(max_iterations),
        confidence=float(confidence),
    )
    if rvec is None or tvec is None:
        return None, None, inliers
    return rvec.reshape(3), tvec.reshape(3), inliers


def refit_pnp_core(
    rvec,
    tvec,
    points3d,
    points2d,
    K,
    std_c,
    std_f,
    mconf,
    params,
    image_size,
    dist=None,
    threshold=12.0,
    max_nfev=100,
):
    del max_nfev
    params = np.asarray(params, dtype=np.float64).reshape(-1)
    if len(params) == 11:
        params = params[:9]
    if len(params) != 9:
        raise ValueError(f"Expected 9 calibration values, got {len(params)}.")

    points3d = np.asarray(points3d, dtype=np.float64)
    points2d = np.asarray(points2d, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64).reshape(3, 3)
    dist = np.zeros(4, dtype=np.float64) if dist is None else np.asarray(dist, dtype=np.float64)
    rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
    tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
    W, H = image_size

    err = reprojection_error_pnp(points2d, points3d, rvec, tvec, K, dist)
    weights = fine_posterior_weights_from_params(
        params,
        std_c,
        std_f,
        mconf,
        err,
        H,
        W,
    )

    rvec, tvec = weighted_refine_pnp(points2d, points3d, weights, K, dist, rvec, tvec)
    err = reprojection_error_pnp(points2d, points3d, rvec, tvec, K, dist)
    final_inliers = np.sum(err * err, axis=1) <= threshold * threshold
    return rvec.reshape(3), tvec.reshape(3), final_inliers, weights


def estimate_pnp_core(
    points3d,
    points2d,
    K,
    std_c,
    std_f,
    mconf,
    params,
    image_size,
    dist=None,
    threshold=12.0,
    confidence=0.999,
    max_iterations=10_000,
):
    rvec, tvec, inliers = estimate_pnp(
        points3d,
        points2d,
        K,
        dist,
        threshold=threshold,
        confidence=confidence,
        max_iterations=max_iterations,
    )
    if rvec is None or tvec is None:
        return None, None, inliers, np.zeros(len(points3d), dtype=np.float64)

    return refit_pnp_core(
        rvec,
        tvec,
        points3d,
        points2d,
        K,
        std_c,
        std_f,
        mconf,
        params,
        image_size,
        dist,
        threshold=threshold,
    )
