from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, Iterable, Sequence

import gsplat
import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import transform_matrix
from PIL import Image
from pyquaternion import Quaternion
from torch.utils.data import Dataset
import wandb
import tempfile
from datetime import datetime
from dotenv import load_dotenv
from geo_forge.train.gs_train_config import GsTrainConfig


load_dotenv()


def _parse_timestamp(stem: str) -> int:
    """Extract the leading timestamp from a filename stem."""
    token = stem.split("_", 1)[0]
    try:
        return int(token)
    except ValueError as exc:
        raise ValueError(
            f"Could not parse timestamp from filename stem '{stem}'. "
            "Expected '<timestamp>_<camera>...'"
        ) from exc


def _to_image_tensor(path: Path) -> tuple[torch.Tensor, int, int]:
    """Load an RGB image and return CHW float tensor in [0, 1] with spatial dims."""
    with Image.open(path) as img:
        rgb = img.convert("RGB")
        width, height = rgb.size
        arr = np.asarray(rgb, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        return tensor, width, height


class RoseNuScenesDataset(Dataset[dict[str, object]]):
    """
    Dataset that pairs ROSE object-removed frames with NuScenes camera poses.

    The loader expects the directory layout produced by ``rose_preprocessor``,
    i.e. ``<dataset_root>/<scene>/<camera>/object_removed_images/object_removed_images/*.png``.
    Filenames must start with the NuScenes timestamp used for the original mask.
    """

    def __init__(
        self,
        version: str | None = None,
        scene_filter: Sequence[str] | None = None,
        camera_filter: Sequence[str] | None = None,
    ) -> None:
        super().__init__()
        dataset_root_env = os.getenv("GEOFORGE_DATASET_ROOT")
        if not dataset_root_env:
            raise EnvironmentError(
                "GEOFORGE_DATASET_ROOT must be set to the directory containing ROSE outputs."
            )
        self.dataset_root = Path(dataset_root_env).expanduser()
        if not self.dataset_root.exists():
            raise FileNotFoundError(
                f"GEOFORGE_DATASET_ROOT path not found at {self.dataset_root}"
            )

        dataroot_env = os.getenv("NUSCENES_DATAROOT")
        if not dataroot_env:
            raise EnvironmentError(
                "NUSCENES_DATAROOT must be set to your NuScenes dataroot."
            )
        self.dataroot = Path(dataroot_env).expanduser()
        if not self.dataroot.exists():
            raise FileNotFoundError(
                f"NuScenes dataroot not found at {self.dataroot}. "
                "Set NUSCENES_DATAROOT to your dataset path."
            )

        nusc_version = version or os.getenv("NUSCENES_VERSION", "v1.0-mini")
        self.nusc = NuScenes(
            version=nusc_version, dataroot=str(self.dataroot), verbose=False
        )
        self.scene_filter = set(scene_filter) if scene_filter else None
        self.camera_filter = (
            {c.lower() for c in camera_filter} if camera_filter else None
        )

        self._pose_index = self._build_pose_index()
        self.samples: list[dict[str, object]] = self._collect_samples()
        if not self.samples:
            raise RuntimeError(
                "No ROSE outputs found. Run the preprocessing pipeline first or "
                "adjust scene/camera filters."
            )

    def _build_pose_index(self) -> dict[tuple[str, str, int], Dict[str, object]]:
        """
        Map (scene, camera, timestamp) -> NuScenes sample_data entry.

        Supports matching by LiDAR timestamp for keyframes to remain compatible with
        mask stems produced from ``iterate_synchronized_samples``.
        """
        scene_name_by_token = {
            scene["token"]: scene["name"] for scene in self.nusc.scene
        }
        sample_token_to_scene: dict[str, str] = {}
        for sample in self.nusc.sample:
            scene_name = scene_name_by_token.get(sample["scene_token"])
            if scene_name is None:
                continue
            if self.scene_filter and scene_name not in self.scene_filter:
                continue
            sample_token_to_scene[sample["token"]] = scene_name

        sample_token_to_lidar_ts: dict[str, int] = {}
        for sample_data in self.nusc.sample_data:
            if not sample_data.get("is_key_frame", True):
                continue
            if sample_data.get("channel") != "LIDAR_TOP":
                continue
            sample_token = sample_data.get("sample_token")
            if sample_token not in sample_token_to_scene:
                continue
            sample_token_to_lidar_ts[sample_token] = int(sample_data["timestamp"])

        pose_index: dict[tuple[str, str, int], Dict[str, object]] = {}
        for sample_data in self.nusc.sample_data:
            channel = sample_data.get("channel", "").lower()
            if not channel.startswith("cam_"):
                continue

            sample_token = sample_data.get("sample_token")
            scene_name = sample_token_to_scene.get(sample_token)
            if scene_name is None:
                continue
            if self.camera_filter and channel not in self.camera_filter:
                continue

            timestamp = int(sample_data["timestamp"])
            pose_index[(scene_name, channel, timestamp)] = sample_data

            if sample_data.get("is_key_frame", True):
                lidar_ts = sample_token_to_lidar_ts.get(sample_token)
                if lidar_ts is not None:
                    pose_index.setdefault((scene_name, channel, lidar_ts), sample_data)

        return pose_index

    def _candidate_image_dirs(self, cam_dir: Path) -> Iterable[Path]:
        yield cam_dir / "object_removed_images" / "object_removed_images"
        yield cam_dir / "object_removed_images"
        yield cam_dir

    def _collect_samples(self) -> list[dict[str, object]]:
        samples: list[dict[str, object]] = []
        for scene_dir in sorted(self.dataset_root.iterdir()):
            if not scene_dir.is_dir():
                continue
            if self.scene_filter and scene_dir.name not in self.scene_filter:
                continue

            for cam_dir in sorted(scene_dir.iterdir()):
                if not cam_dir.is_dir():
                    continue

                cam_name = cam_dir.name.lower()
                if self.camera_filter and cam_name not in self.camera_filter:
                    continue

                image_dir = next(
                    (d for d in self._candidate_image_dirs(cam_dir) if d.exists()), None
                )
                if image_dir is None:
                    continue

                image_paths = sorted(
                    [
                        *image_dir.glob("*.png"),
                        *image_dir.glob("*.jpg"),
                        *image_dir.glob("*.jpeg"),
                    ]
                )
                for image_path in image_paths:
                    timestamp = _parse_timestamp(image_path.stem)
                    pose_meta = self._pose_index.get(
                        (scene_dir.name, cam_name, timestamp)
                    )
                    if pose_meta is None:
                        # Skip silently to keep the sample code lightweight.
                        continue

                    intrinsics, c2w = self._camera_from_sample_data(pose_meta)
                    image_tensor, width, height = _to_image_tensor(image_path)
                    samples.append(
                        {
                            "image": image_tensor,
                            "intrinsics": intrinsics,
                            "c2w": c2w,
                            "width": width,
                            "height": height,
                            "scene": scene_dir.name,
                            "camera": cam_name,
                            "timestamp": timestamp,
                        }
                    )
        return samples

    def _camera_from_sample_data(
        self, sample_data: Dict[str, object]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        calibrated = self.nusc.get(
            "calibrated_sensor", sample_data["calibrated_sensor_token"]
        )
        ego_pose = self.nusc.get("ego_pose", sample_data["ego_pose_token"])

        cam_to_ego = transform_matrix(
            calibrated["translation"],
            Quaternion(calibrated["rotation"]),
            inverse=False,
        )
        ego_to_world = transform_matrix(
            ego_pose["translation"],
            Quaternion(ego_pose["rotation"]),
            inverse=False,
        )
        cam_to_world = ego_to_world @ cam_to_ego

        intrinsics = torch.tensor(
            np.asarray(calibrated["camera_intrinsic"], dtype=np.float32)
        )
        c2w = torch.tensor(cam_to_world, dtype=torch.float32)
        return intrinsics, c2w

    def __len__(self) -> int:
        return len(self.samples)


def __getitem__(self, idx: int) -> dict[str, object]:
    return self.samples[idx]


def _tensor_to_uint8_image(tensor: torch.Tensor) -> np.ndarray:
    """Convert CHW float tensor in [0, 1] to HWC uint8."""
    img = torch.clamp(tensor, 0.0, 1.0).detach().cpu()
    if img.dim() != 3 or img.shape[0] != 3:
        raise ValueError(
            f"Expected CHW tensor with 3 channels, got shape {tuple(img.shape)}"
        )
    return (img.permute(1, 2, 0).numpy() * 255.0).astype(np.uint8)


def _stack_camera_grid(
    preds: Sequence[torch.Tensor],
    gts: Sequence[torch.Tensor],
    cameras: Sequence[str],
    columns: int = 3,
) -> np.ndarray:
    """Arrange (pred, gt) pairs into a grid for visualization."""
    if len(preds) != len(gts) or len(preds) != len(cameras):
        raise ValueError("Preds, GTs, and cameras must have equal length.")

    cells: list[np.ndarray] = []
    for pred, gt in zip(preds, gts):
        pred_img = _tensor_to_uint8_image(pred)
        gt_img = _tensor_to_uint8_image(gt)
        if pred_img.shape[:2] != gt_img.shape[:2]:
            raise ValueError(
                "Predicted and GT images must share spatial size for visualization."
            )
        cell = np.concatenate([pred_img, gt_img], axis=0)
        cells.append(cell)

    if not cells:
        raise ValueError("No images provided for grid stacking.")

    cell_h, cell_w, _ = cells[0].shape
    rows = int(np.ceil(len(cells) / columns))
    canvas = np.zeros((rows * cell_h, columns * cell_w, 3), dtype=np.uint8)
    for idx, cell in enumerate(cells):
        r, c = divmod(idx, columns)
        y0, y1 = r * cell_h, (r + 1) * cell_h
        x0, x1 = c * cell_w, (c + 1) * cell_w
        canvas[y0:y1, x0:x1] = cell

    # Overlay camera labels in the top-left corner of each cell (small white strip).
    for idx, cam in enumerate(cameras):
        r, c = divmod(idx, columns)
        y0, x0 = r * cell_h, c * cell_w
        strip_h = min(18, cell_h // 12)
        canvas[y0 : y0 + strip_h, x0 : x0 + min(len(cam) * 8 + 8, cell_w), :] = 255
    return canvas


class GaussianSplattingModel(torch.nn.Module):
    """Minimal Gaussian parameter container with a gsplat render helper."""

    def __init__(self, num_gaussians: int, device: torch.device) -> None:
        super().__init__()
        self.means = torch.nn.Parameter(
            torch.randn(num_gaussians, 3, device=device) * 0.5
        )
        self.scales = torch.nn.Parameter(
            torch.full((num_gaussians, 3), 0.1, device=device)
        )
        self.rotations = torch.nn.Parameter(
            torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).repeat(num_gaussians, 1)
        )
        self.opacities = torch.nn.Parameter(
            torch.full((num_gaussians, 1), 0.5, device=device)
        )
        self.colors = torch.nn.Parameter(torch.rand(num_gaussians, 3, device=device))

    def render(
        self,
        intrinsics: torch.Tensor,
        c2w: torch.Tensor,
        width: int,
        height: int,
        background: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Render a single view using gsplat.

        gsplat expects world-to-camera view matrices; the NuScenes poses are
        camera-to-world, so we invert before rendering.
        """
        viewmat = torch.inverse(c2w)[None]  # (1, 4, 4)
        Ks = intrinsics[None]  # (1, 3, 3)
        bg = (
            background.to(self.means.device)
            if background is not None
            else torch.zeros(3, device=self.means.device)
        )

        render_out = gsplat.render(
            viewmats=viewmat,
            Ks=Ks,
            width=width,
            height=height,
            means3d=self.means,
            scales=self.scales,
            rotations=self.rotations,
            opacities=self.opacities,
            colors=self.colors,
            backgrounds=bg,
        )
        rendered = (
            render_out[0] if isinstance(render_out, (list, tuple)) else render_out
        )
        if rendered.dim() == 4:
            rendered = rendered.permute(0, 3, 1, 2)
        return rendered[0]


def _build_eval_sets(
    dataset: RoseNuScenesDataset,
    max_sets: int = 2,
    target_cameras: Sequence[str] | None = None,
) -> list[list[dict[str, object]]]:
    """
    Group samples by (scene, timestamp) to render multiple camera views together.
    """
    target = [c.lower() for c in target_cameras] if target_cameras else None
    groups: dict[tuple[str, int], dict[str, dict[str, object]]] = {}
    for sample in dataset.samples:
        cam = str(sample["camera"])
        if target and cam not in target:
            continue
        key = (str(sample["scene"]), int(sample["timestamp"]))
        groups.setdefault(key, {})
        groups[key][cam] = sample

    eval_sets: list[list[dict[str, object]]] = []
    for (_, _), cam_map in groups.items():
        if target:
            ordered = [cam_map[c] for c in target if c in cam_map]
        else:
            ordered = [cam_map[c] for c in sorted(cam_map.keys())]
        if ordered:
            eval_sets.append(ordered)
        if len(eval_sets) >= max_sets:
            break
    return eval_sets


def _render_eval_set(
    model: GaussianSplattingModel,
    camera_set: Sequence[dict[str, object]],
    device: torch.device,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[str]]:
    preds: list[torch.Tensor] = []
    gts: list[torch.Tensor] = []
    cams: list[str] = []
    for sample in camera_set:
        intrinsics = sample["intrinsics"].to(device)
        c2w = sample["c2w"].to(device)
        width = int(sample["width"])
        height = int(sample["height"])
        preds.append(
            model.render(intrinsics=intrinsics, c2w=c2w, width=width, height=height)
        )
        gts.append(sample["image"].to(device))
        cams.append(str(sample["camera"]))
    return preds, gts, cams


def _log_wandb_render_gif(
    model: GaussianSplattingModel,
    eval_sets: Sequence[Sequence[dict[str, object]]],
    device: torch.device,
    history_frames: list[np.ndarray],
    max_history: int,
    step: int,
) -> None:
    if not eval_sets:
        return

    preds, gts, cams = _render_eval_set(model, eval_sets[0], device=device)
    grid = _stack_camera_grid(preds, gts, cams)

    history_frames.append(grid)
    if len(history_frames) > max_history:
        del history_frames[0 : len(history_frames) - max_history]

    with tempfile.NamedTemporaryFile(suffix=".gif", delete=False) as tmp:
        gif_path = Path(tmp.name)
    iio.imwrite(gif_path, history_frames, fps=2, loop=0)
    wandb.log(
        {"render_gif": wandb.Video(str(gif_path), fps=2, format="gif")}, step=step
    )
    gif_path.unlink(missing_ok=True)


def train_gaussian_splatting(
    dataset: RoseNuScenesDataset,
    config: GsTrainConfig,
) -> None:
    """
    Lightweight training loop that optimizes Gaussian parameters against ROSE frames.
    """
    if len(dataset) == 0:
        raise ValueError("Dataset is empty; nothing to train on.")

    device_t = torch.device(
        config.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model = GaussianSplattingModel(num_gaussians=config.num_gaussians, device=device_t)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    use_wandb = bool(config.wandb_project)
    if use_wandb:
        run_name = (
            config.wandb_run_name
            if config.wandb_run_name is not None
            else datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
        )
        wandb.init(
            project=config.wandb_project,
            name=run_name,
            config={
                "num_steps": config.steps,
                "num_gaussians": config.num_gaussians,
                "lr": config.lr,
                "log_every": config.log_interval,
                "log_render_every": config.render_interval,
            },
        )

    eval_sets = _build_eval_sets(
        dataset,
        max_sets=config.max_eval_sets,
        target_cameras=config.cameras,
    )
    render_history: list[np.ndarray] = []
    render_interval = (
        config.render_interval
        if config.render_interval is not None
        else config.log_interval
    )

    for step in range(config.steps):
        sample = dataset[step % len(dataset)]
        image = sample["image"].to(device_t)  # (3, H, W)
        intrinsics = sample["intrinsics"].to(device_t)
        c2w = sample["c2w"].to(device_t)
        width = int(sample["width"])
        height = int(sample["height"])

        pred = model.render(intrinsics=intrinsics, c2w=c2w, width=width, height=height)
        loss = F.mse_loss(pred, image)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % config.log_interval == 0:
            scene = sample["scene"]
            camera = sample["camera"]
            timestamp = sample["timestamp"]
            print(
                f"[step {step + 1:04d}] "
                f"loss={loss.item():.4f} scene={scene} cam={camera} ts={timestamp}"
            )
            if use_wandb:
                wandb.log({"loss": loss.item()}, step=step + 1)

        if use_wandb and render_interval and (step + 1) % render_interval == 0:
            _log_wandb_render_gif(
                model=model,
                eval_sets=eval_sets,
                device=device_t,
                history_frames=render_history,
                max_history=config.max_render_history,
                step=step + 1,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a Gaussian splatting demo using ROSE outputs and NuScenes poses."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to a YAML file containing GsTrainConfig values.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = GsTrainConfig.from_yaml(args.config)
    dataset = RoseNuScenesDataset(
        scene_filter=config.scenes,
        camera_filter=config.cameras,
    )
    train_gaussian_splatting(dataset, config=config)


if __name__ == "__main__":
    main()
