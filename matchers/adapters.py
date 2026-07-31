"""Inference-only adapters exposing uncertainty cues from supported matchers."""

from __future__ import annotations

import copy
from pathlib import Path
import sys
import time
from typing import Any

import cv2
import numpy as np
import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
THIRD_PARTY_ROOT = REPOSITORY_ROOT / "third_party"
WEIGHTS_ROOT = REPOSITORY_ROOT / "weights"

DEFAULT_WEIGHTS = {
    "edm": WEIGHTS_ROOT / "edm_outdoor.ckpt",
    "eloftr": WEIGHTS_ROOT / "eloftr_outdoor.ckpt",
    "loftr": WEIGHTS_ROOT / "loftr_outdoor.ckpt",
    "gim_loftr": WEIGHTS_ROOT / "gim_loftr_50h.ckpt",
    "comatch": WEIGHTS_ROOT / "comatch_outdoor.ckpt",
    "xfeat": WEIGHTS_ROOT / "xfeat.pt",
}

ALIASES = {
    "efficientloftr": "eloftr",
    "gim-loftr": "gim_loftr",
    "gimloftr": "gim_loftr",
}


def canonical_name(name: str) -> str:
    name = name.lower()
    return ALIASES.get(name, name)


def _activate(repository: str, namespace: str) -> Path:
    """Activate one vendored repository that uses a top-level import namespace."""

    root = THIRD_PARTY_ROOT / repository
    if not root.is_dir():
        raise FileNotFoundError(f"Missing vendored matcher source: {root}")
    for candidate in THIRD_PARTY_ROOT.iterdir():
        try:
            sys.path.remove(str(candidate))
        except ValueError:
            pass
    for module_name in list(sys.modules):
        if module_name == namespace or module_name.startswith(f"{namespace}."):
            del sys.modules[module_name]
    sys.path.insert(0, str(root))
    return root


def _lower_config(value: Any) -> Any:
    if hasattr(value, "items"):
        return {str(key).lower(): _lower_config(item) for key, item in value.items()}
    return value


def _load_checkpoint(path: Path, device: torch.device) -> Any:
    if not path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            "Download it as described in README.md or pass weights=... explicitly."
        )
    return torch.load(path, map_location=device)


def _to_numpy(batch: dict[str, Any], key: str) -> np.ndarray | None:
    value = batch.get(key)
    if value is None:
        return None
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _gray_tensor(image: np.ndarray | torch.Tensor, device: torch.device) -> torch.Tensor:
    if torch.is_tensor(image):
        tensor = image
        if tensor.ndim == 2:
            tensor = tensor[None, None]
        elif tensor.ndim == 3:
            tensor = tensor[None]
        if tensor.shape[1] == 3:
            tensor = tensor.mean(dim=1, keepdim=True)
        tensor = tensor.to(device).float()
        return tensor / 255.0 if tensor.max() > 2 else tensor

    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return torch.from_numpy(image)[None, None].to(device).float() / 255.0


def _rgb_tensor(
    image: np.ndarray | torch.Tensor, device: torch.device, divisor: int = 8
) -> tuple[torch.Tensor, torch.Tensor]:
    if torch.is_tensor(image):
        tensor = image
        if tensor.ndim == 3:
            tensor = tensor[None]
        if tensor.shape[1] == 1:
            tensor = tensor.repeat(1, 3, 1, 1)
        tensor = tensor[:, :3].to(device).float()
        if tensor.max() > 2:
            tensor = tensor / 255.0
    else:
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        tensor = (
            torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))[None]
            .to(device)
            .float()
            / 255.0
        )
    old_height, old_width = tensor.shape[-2:]
    new_height = max(divisor, old_height // divisor * divisor)
    new_width = max(divisor, old_width // divisor * divisor)
    if (new_height, new_width) != (old_height, old_width):
        tensor = torch.nn.functional.interpolate(
            tensor,
            size=(new_height, new_width),
            mode="bilinear",
            align_corners=False,
        )
    scale = tensor.new_tensor(
        [[old_width / new_width, old_height / new_height]]
    ).repeat(tensor.shape[0], 1)
    return tensor, scale


def _result(batch: dict[str, Any], elapsed: float) -> dict[str, Any]:
    points0 = _to_numpy(batch, "mkpts0_f")
    points1 = _to_numpy(batch, "mkpts1_f")
    if points0 is None or points1 is None:
        raise RuntimeError("Matcher did not return mkpts0_f and mkpts1_f.")
    matches = (
        np.concatenate((points0, points1), axis=1)
        if len(points0)
        else np.empty((0, 4), dtype=np.float32)
    )
    output = {
        "matches": matches.astype(np.float32, copy=False),
        "confidence": _to_numpy(batch, "mconf"),
        "std_coarse": _to_numpy(batch, "std_coarse"),
        "std_fine": _to_numpy(batch, "std_fine"),
        "elapsed": float(elapsed),
        "uncertainty_elapsed": float(batch.get("uncertainty_time", 0.0)),
    }
    for key in ("confidence", "std_coarse", "std_fine"):
        if output[key] is None:
            raise RuntimeError(
                f"Matcher did not expose {key}. Ensure calibration/uncertainty mode is enabled."
            )
    return output


class _Matcher:
    def __init__(self, device: str | torch.device | None = None) -> None:
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)


class EDMMatcher(_Matcher):
    def __init__(
        self,
        weights: Path,
        device: str | torch.device | None = None,
        confidence_threshold: float = 0.2,
    ) -> None:
        super().__init__(device)
        root = _activate("EDM", "src")
        from src.config.default import get_cfg_defaults
        from src.edm import EDM

        config = get_cfg_defaults()
        config.merge_from_file(root / "configs/edm/outdoor/edm_base.py")
        config.merge_from_file(root / "configs/data/megadepth_test_1500.py")
        config.EDM.COARSE.MCONF_THR = confidence_threshold
        config.EDM.COARSE.BORDER_RM = 2
        config.EDM.COARSE.CALIBRATION = True
        self.model = EDM(config=_lower_config(config)["edm"]).to(self.device).eval()
        checkpoint = _load_checkpoint(weights, self.device)
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)

    def match(self, image0: np.ndarray, image1: np.ndarray) -> dict[str, Any]:
        batch = {
            "image0": _gray_tensor(image0, self.device),
            "image1": _gray_tensor(image1, self.device),
        }
        self._synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            self.model(batch)
        self._synchronize()
        return _result(batch, time.perf_counter() - start)


class EfficientLoFTRMatcher(_Matcher):
    def __init__(
        self, weights: Path, device: str | torch.device | None = None
    ) -> None:
        super().__init__(device)
        _activate("EfficientLoFTR", "src")
        from src.loftr import LoFTR, full_default_cfg, reparameter

        config = copy.deepcopy(full_default_cfg)
        config["mp"] = self.device.type == "cuda"
        config["coarse"]["no_flash"] = True
        config["match_coarse"]["calibration"] = True
        model = LoFTR(config=config)
        checkpoint = _load_checkpoint(weights, self.device)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        self.model = reparameter(model).to(self.device).eval()

    def match(self, image0: np.ndarray, image1: np.ndarray) -> dict[str, Any]:
        batch = {
            "image0": _gray_tensor(image0, self.device),
            "image1": _gray_tensor(image1, self.device),
        }
        self._synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            if self.device.type == "cuda":
                with torch.autocast(device_type="cuda"):
                    self.model(batch)
            else:
                self.model(batch)
        self._synchronize()
        return _result(batch, time.perf_counter() - start)


class LoFTRMatcher(_Matcher):
    def __init__(
        self, weights: Path, device: str | torch.device | None = None
    ) -> None:
        super().__init__(device)
        _activate("LoFTR", "src")
        from src.loftr import LoFTR, default_cfg

        config = copy.deepcopy(default_cfg)
        config["match_coarse"]["calibration"] = True
        self.model = LoFTR(config=config).to(self.device).eval()
        checkpoint = _load_checkpoint(weights, self.device)
        state = checkpoint.get("state_dict", checkpoint)
        self.model.load_state_dict(state, strict=True)

    def match(self, image0: np.ndarray, image1: np.ndarray) -> dict[str, Any]:
        batch = {
            "image0": _gray_tensor(image0, self.device),
            "image1": _gray_tensor(image1, self.device),
        }
        self._synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            self.model(batch)
        self._synchronize()
        return _result(batch, time.perf_counter() - start)


class GIMLoFTRMatcher(_Matcher):
    def __init__(
        self,
        weights: Path,
        device: str | torch.device | None = None,
        confidence_threshold: float = 0.2,
    ) -> None:
        super().__init__(device)
        _activate("gim", "networks")
        from networks.loftr.config import get_cfg_defaults
        from networks.loftr.loftr import LoFTR

        config = _lower_config(get_cfg_defaults())["loftr"]
        config["match_coarse"]["thr"] = confidence_threshold
        config["match_coarse"]["calibration"] = True
        self.model = LoFTR(config).to(self.device).eval()
        checkpoint = _load_checkpoint(weights, self.device)
        state = checkpoint.get("state_dict", checkpoint)
        self.model.load_state_dict(state, strict=True)

    def match(self, image0: np.ndarray, image1: np.ndarray) -> dict[str, Any]:
        color0, scale0 = _rgb_tensor(image0, self.device)
        color1, scale1 = _rgb_tensor(image1, self.device)
        batch = {
            "color0": color0,
            "color1": color1,
            "image0": color0,
            "image1": color1,
            "scale0": scale0,
            "scale1": scale1,
        }
        self._synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            self.model(batch)
        self._synchronize()
        return _result(batch, time.perf_counter() - start)


class CoMatchMatcher(_Matcher):
    def __init__(
        self, weights: Path, device: str | torch.device | None = None
    ) -> None:
        super().__init__(device)
        root = _activate("CoMatch", "src")
        from src.config.default import get_cfg_defaults
        from src.loftr import LoFTR

        config = get_cfg_defaults()
        config.merge_from_file(root / "configs/loftr/comatch_full.py")
        config.LOFTR.COARSE.NPE = [832, 832, 832, 832]
        config.LOFTR.MATCH_COARSE.CALIBRATION = True
        self.model = LoFTR(config=_lower_config(config.LOFTR)).to(self.device).eval()
        checkpoint = _load_checkpoint(weights, self.device)
        self.model.load_state_dict(checkpoint["state_dict"], strict=True)

    def match(self, image0: np.ndarray, image1: np.ndarray) -> dict[str, Any]:
        batch = {
            "image0": _gray_tensor(image0, self.device),
            "image1": _gray_tensor(image1, self.device),
        }
        self._synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            self.model(batch)
        self._synchronize()
        return _result(batch, time.perf_counter() - start)


class XFeatMatcher(_Matcher):
    def __init__(
        self, weights: Path, device: str | torch.device | None = None
    ) -> None:
        super().__init__(device)
        _activate("accelerated_features", "modules")
        from modules.model import XFeatModel
        from modules.xfeat import XFeat

        model = XFeatModel()
        checkpoint = _load_checkpoint(weights, self.device)
        state = checkpoint.get("model", checkpoint)
        compatible = {
            key: value
            for key, value in state.items()
            if key in model.state_dict() and model.state_dict()[key].shape == value.shape
        }
        model.load_state_dict(compatible, strict=False)
        self.model = XFeat(weights=None, calibration=True).to(self.device).eval()
        self.model.net = model.to(self.device).eval()

    def match(self, image0: np.ndarray, image1: np.ndarray) -> dict[str, Any]:
        tensor0 = _gray_tensor(image0, self.device)
        tensor1 = _gray_tensor(image1, self.device)
        self._synchronize()
        start = time.perf_counter()
        with torch.inference_mode():
            # XFeat* refines its first input. Reverse the call, then restore the
            # public source->target coordinate convention.
            matches, std_coarse, std_fine, confidence = self.model.match_xfeat_star(
                tensor1, tensor0
            )
        self._synchronize()
        matches = torch.cat((matches[:, 2:4], matches[:, :2]), dim=1)
        return {
            "matches": matches.detach().cpu().numpy(),
            "confidence": confidence.detach().cpu().numpy(),
            "std_coarse": std_coarse.detach().cpu().numpy(),
            "std_fine": std_fine.detach().cpu().numpy(),
            "elapsed": float(time.perf_counter() - start),
            "uncertainty_elapsed": float(
                getattr(self.model, "last_uncertainty_time", 0.0)
            ),
        }


def load_matcher(
    name: str,
    *,
    weights: str | Path | None = None,
    device: str | torch.device | None = None,
    confidence_threshold: float = 0.2,
) -> _Matcher:
    """Construct one supported uncertainty-enabled matcher."""

    name = canonical_name(name)
    if name not in DEFAULT_WEIGHTS:
        available = ", ".join(sorted(DEFAULT_WEIGHTS))
        raise ValueError(f"Unknown matcher '{name}'. Available: {available}")
    weight_path = Path(weights) if weights is not None else DEFAULT_WEIGHTS[name]
    classes = {
        "edm": EDMMatcher,
        "eloftr": EfficientLoFTRMatcher,
        "loftr": LoFTRMatcher,
        "gim_loftr": GIMLoFTRMatcher,
        "comatch": CoMatchMatcher,
        "xfeat": XFeatMatcher,
    }
    matcher_class = classes[name]
    if name in {"edm", "gim_loftr"}:
        return matcher_class(
            weight_path,
            device=device,
            confidence_threshold=confidence_threshold,
        )
    return matcher_class(weight_path, device=device)
