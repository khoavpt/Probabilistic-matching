from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys

import numpy as np
from scipy.optimize import minimize


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


PARAMETER_NAMES = ("a1", "a2", "b1", "b2", "k_s", "k_m", "t1", "t2", "t_m")
MATCHER_ALIASES = {
    "efficientloftr": "eloftr",
    "gimloftr": "gim_loftr",
}


def canonical_matcher_name(name):
    name = name.lower()
    return MATCHER_ALIASES.get(name, name)


def normalize_coarse_std(std_c, W, H):
    image_size = np.array([W, H], dtype=np.float64)
    return np.asarray(std_c, dtype=np.float64) / image_size * 100.0


def confidence_risk(mconf):
    return -np.log(np.clip(np.asarray(mconf, dtype=np.float64), 1e-8, 1.0))


def gate_alpha(std_c_norm_d, mconf, k_s, k_m, t_s, t_m):
    z = k_s * (std_c_norm_d - t_s) + k_m * (confidence_risk(mconf) - t_m)
    z = np.clip(z, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-z))


def mixture_nll_loss(params, std_c, std_f, mconf, err, W, H, alpha_l1=0.0):
    std_c = np.asarray(std_c, dtype=np.float64)
    std_f = np.asarray(std_f, dtype=np.float64)
    mconf = np.asarray(mconf, dtype=np.float64).reshape(-1)
    err = np.asarray(err, dtype=np.float64)
    a1, a2, b1, b2, k_s, k_m, t1, t2, t_m = np.asarray(
        params, dtype=np.float64
    )
    a = [a1, a2]
    b = [b1, b2]
    t = [t1, t2]
    std_c_norm = normalize_coarse_std(std_c, W, H)

    loss = 0.0
    alpha_penalty = 0.0
    for d in range(2):
        alpha = gate_alpha(std_c_norm[:, d], mconf, k_s, k_m, t[d], t_m)
        alpha_penalty += alpha.mean()

        scale_fine = np.maximum(np.sqrt(b[d]) * std_f[:, d], 1e-6)
        scale_coarse = np.maximum(np.sqrt(a[d]) * std_c[:, d], 1e-6)

        log_p_fine = -np.abs(err[:, d]) / scale_fine - np.log(2.0 * scale_fine)
        log_p_coarse = -np.abs(err[:, d]) / scale_coarse - np.log(2.0 * scale_coarse)
        log_mix = np.logaddexp(
            np.log(1.0 - alpha + 1e-8) + log_p_fine,
            np.log(alpha + 1e-8) + log_p_coarse,
        )
        loss += -np.mean(log_mix)

    return float(loss / 2.0 + alpha_l1 * alpha_penalty / 2.0)


def fit_calibration_params(
    std_c,
    std_f,
    mconf,
    err,
    W,
    H,
    alpha_l1=0.15,
    max_iterations=120,
):
    std_c = np.asarray(std_c, dtype=np.float64)
    std_f = np.asarray(std_f, dtype=np.float64)
    mconf = np.asarray(mconf, dtype=np.float64).reshape(-1)
    err = np.asarray(err, dtype=np.float64)

    std_c_norm = normalize_coarse_std(std_c, W, H)
    mconf_risk = confidence_risk(mconf)

    result = minimize(
        lambda p: mixture_nll_loss(
            p, std_c, std_f, mconf, err, W, H, alpha_l1=alpha_l1
        ),
        x0=[
            1.0,
            1.0,
            20.0,
            20.0,
            0.50,
            0.50,
            np.percentile(std_c_norm[:, 0], 75),
            np.percentile(std_c_norm[:, 1], 75),
            np.percentile(mconf_risk, 75),
        ],
        bounds=[
            (0.01, 100.0),
            (0.01, 100.0),
            (0.01, 100.0),
            (0.01, 100.0),
            (0.001, 10.0),
            (0.0, 10.0),
            (np.percentile(std_c_norm[:, 0], 1), np.percentile(std_c_norm[:, 0], 99)),
            (np.percentile(std_c_norm[:, 1], 1), np.percentile(std_c_norm[:, 1], 99)),
            (np.percentile(mconf_risk, 1), np.percentile(mconf_risk, 99)),
        ],
        options={"maxiter": int(max_iterations), "ftol": 1e-7, "eps": 1e-2},
        method="powell",
    )
    return np.asarray(result.x, dtype=np.float64), result


def print_uncertainty_fit_inputs(name, std_c, std_f, mconf, err):
    abs_err = np.abs(err)

    def pct(x):
        return np.percentile(x, [1, 10, 50, 90, 99])

    def fmt(vals):
        return "[" + ", ".join(f"{v:.3f}" for v in vals) + "]"

    print(f"\n{name} calibration input sanity")
    for d, axis in enumerate(["x", "y"]):
        ratio_f = abs_err[:, d] / np.maximum(std_f[:, d], 1e-6)
        ratio_c = abs_err[:, d] / np.maximum(std_c[:, d], 1e-6)
        print(f"  {axis}: |err| pct        {fmt(pct(abs_err[:, d]))}")
        print(f"  {axis}: std_f pct       {fmt(pct(std_f[:, d]))}")
        print(f"  {axis}: std_c pct       {fmt(pct(std_c[:, d]))}")
        print(f"  {axis}: |err|/std_f pct {fmt(pct(ratio_f))}")
        print(f"  {axis}: |err|/std_c pct {fmt(pct(ratio_c))}")
    print(f"  mconf pct {fmt(pct(mconf))}\n")


def _csv(value):
    if value is None:
        return None
    items = []
    for chunk in value:
        items.extend(part.strip() for part in chunk.split(",") if part.strip())
    return items or None


def build_dataset(args):
    from torch.utils.data import ConcatDataset

    if args.dataset == "megadepth":
        from datasets.megadepth_minima_v2 import MegadepthBuilder

        builder = MegadepthBuilder(
            data_root=args.data_root,
            modality_list=["visible"],
        )
        scenes = builder.build_scenes(
            RANK=args.rank,
            split=args.split,
            min_overlap=args.min_overlap,
            scene_names=_csv(args.scene),
            ht=args.height,
            wt=args.width,
        )
    else:
        from datasets.mtv import MTVBuilder

        builder = MTVBuilder(data_root=args.data_root)
        build_kwargs = {
            "RANK": args.rank,
            "split": args.split,
            "min_overlap": args.min_overlap,
            "ht": args.height,
            "wt": args.width,
        }
        scene_names = _csv(args.scene)
        if scene_names is not None:
            build_kwargs["scene_names"] = scene_names
        scenes = builder.build_scenes(**build_kwargs)

    return ConcatDataset(scenes)


def extract_samples(args):
    import torch
    from torch.utils.data import DataLoader, WeightedRandomSampler
    import tqdm

    from datasets import megadepth_warper
    from matchers import load_matcher

    seed = int(args.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    matcher = load_matcher(
        args.matcher,
        weights=args.weights,
        device=device,
        confidence_threshold=args.confidence_threshold,
    )

    dataset = build_dataset(args)
    num_pairs = len(dataset)
    num_samples = num_pairs if args.samples <= 0 else min(args.samples, num_pairs)
    sampler = WeightedRandomSampler(
        torch.ones(num_pairs), replacement=False, num_samples=num_samples
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        sampler=sampler,
        pin_memory=(device.type == "cuda"),
    )

    print(
        f"Extracting {num_samples}/{num_pairs} {args.dataset} pairs at "
        f"{args.width}x{args.height} with matcher={args.matcher}"
    )

    X_coarse = []
    X_fine = []
    X_mconf = []
    y = []
    image_size = None

    for data in tqdm.tqdm(loader, total=len(loader)):
        image0 = data["image0"].to(device)
        image1 = data["image1"].to(device)
        height, width = int(image0.shape[-2]), int(image0.shape[-1])
        if image_size is None:
            image_size = (width, height)

        result = matcher.match(image0, image1)
        matches = np.asarray(result["matches"], dtype=np.float32)
        if len(matches) < args.min_matches:
            continue

        std_coarse = np.asarray(result["std_coarse"], dtype=np.float64)
        std_fine = np.asarray(result["std_fine"], dtype=np.float64)
        mconf = np.asarray(result["confidence"], dtype=np.float64).reshape(-1)
        x1 = torch.from_numpy(matches[:, :2]).to(device=device, dtype=torch.float64)
        with torch.no_grad():
            mask, x2_gt = megadepth_warper.warp_kpts_v2(
                x1.unsqueeze(0),
                data["depth0"].to(device).double(),
                data["depth1"].to(device).double(),
                data["T_0to1"].to(device).double(),
                data["K0"].to(device).double(),
                data["K1"].to(device).double(),
            )

        mask = mask[0].detach().cpu().numpy().astype(bool)
        x2_gt = x2_gt[0].detach().cpu().numpy()
        x2_pred = matches[:, 2:4]
        if mask.sum() < args.min_matches:
            continue

        X_coarse.append(std_coarse[mask])
        X_fine.append(std_fine[mask])
        X_mconf.append(mconf[mask])
        y.append(x2_pred[mask] - x2_gt[mask])

    std_c = np.concatenate(X_coarse)
    std_f = np.concatenate(X_fine)
    mconf = np.concatenate(X_mconf)
    err = np.concatenate(y)

    return std_c, std_f, mconf, err, image_size


def save_result(args, params, result, err, image_size, source):
    output = Path(args.output or f"outputs/calibration/{args.matcher}.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "parameter_order": list(PARAMETER_NAMES),
        "matcher": args.matcher,
        "params": [float(value) for value in params],
        "success": bool(result.success),
        "message": str(result.message),
        "nll": float(result.fun),
        "num_correspondences": int(len(err)),
        "image_size": [int(image_size[0]), int(image_size[1])],
        "gate_penalty": float(args.gate_penalty),
        "source": source,
    }
    with output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    print(f"Saved calibration to {output}")

    if args.params_json:
        params_path = Path(args.params_json)
        if params_path.exists():
            with params_path.open("r", encoding="utf-8") as handle:
                aggregate = json.load(handle)
        else:
            aggregate = {"parameter_order": list(PARAMETER_NAMES), "matchers": {}}
        aggregate.setdefault("parameter_order", list(PARAMETER_NAMES))
        aggregate.setdefault("matchers", {})
        aggregate["matchers"][args.matcher] = [float(value) for value in params]
        params_path.parent.mkdir(parents=True, exist_ok=True)
        with params_path.open("w", encoding="utf-8") as handle:
            json.dump(aggregate, handle, indent=2)
            handle.write("\n")
        print(f"Updated aggregate params in {params_path}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract residuals from a depth dataset and fit CoRe calibration."
    )
    parser.add_argument("--data-root", required=True, help="MegaDepth or MTV root.")
    parser.add_argument("--dataset", choices=["megadepth", "mtv"], default="megadepth")
    parser.add_argument("--matcher", default="eloftr")
    parser.add_argument("--weights", help="Override matcher checkpoint path.")
    parser.add_argument("--output", help="Destination JSON. Defaults to outputs/calibration/<matcher>.json.")
    parser.add_argument("--params-json", help="Update to params.json")
    parser.add_argument("--device", help="cuda, cuda:0, or cpu. Defaults to CUDA when available.")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--split", default="test")
    parser.add_argument("--samples", type=int, default=200, help="Pairs to sample; <=0 means all pairs.")
    parser.add_argument("--min-matches", type=int, default=10)
    parser.add_argument("--min-overlap", type=float, default=0.0)
    parser.add_argument("--scene", action="append", help="Scene name; repeat or comma-separate.")
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--confidence-threshold", type=float, default=0.2)
    parser.add_argument("--gate-penalty", type=float, default=0.0)
    parser.add_argument("--max-iterations", type=int, default=120)
    return parser.parse_args()


def main():
    args = parse_args()
    args.matcher = canonical_matcher_name(args.matcher)
    std_c, std_f, mconf, err, image_size = extract_samples(args)
    source = {
        "type": "dataset",
        "dataset": args.dataset,
        "data_root": args.data_root,
        "split": args.split,
        "samples": int(args.samples),
    }

    print_uncertainty_fit_inputs(args.matcher, std_c, std_f, mconf, err)
    print("Start optimizing")
    params, result = fit_calibration_params(
        std_c,
        std_f,
        mconf,
        err,
        image_size[0],
        image_size[1],
        alpha_l1=args.gate_penalty,
        max_iterations=args.max_iterations,
    )
    values = ", ".join(
        f"{name}={value:.4f}" for name, value in zip(PARAMETER_NAMES, params)
    )
    print(values)
    print(f"Total calibration correspondences: {len(err)}")
    print(f"Optimization success={result.success}, nll={float(result.fun):.6f}")
    save_result(args, params, result, err, image_size, source)


if __name__ == "__main__":
    main()
