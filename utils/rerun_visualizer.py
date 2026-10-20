# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import shutil
import uuid
from pathlib import Path
from typing import Optional, Tuple, Union

import ipdb
import matplotlib
import numpy as np
import rerun
import tempfile

import torch

from iopath.common.file_io import PathManager
from scipy.spatial.transform import Rotation as R

from .common_utils import batch_project, batch_unproject, batch_unproject_uv_space

path_manager = PathManager()

st = ipdb.set_trace


def setup_visualizer(
    app_name: str = "Data Visualization",
    web_port: int = 9091,
    ws_port: int = 9878,
    server_memory_limit: str = "25%",
    open_browser=False,
    serve=True,
) -> None:
    rerun.init(app_name, spawn=False, recording_id="rerun_visualizer")
    if serve:
        rerun.serve(
            open_browser=open_browser,
            web_port=web_port,
            ws_port=ws_port,
            default_blueprint=None,
            recording=None,
            server_memory_limit=server_memory_limit,
        )


def save_recording(
    path,
) -> None:
    path = Path(path)
    # `path_manager.mkdirs()` creates the given URI as a directory. Passing the
    # full `.rrd` path therefore makes basename a folder and `rerun.save`
    # fails with os error 21 (Is a directory).
    path_manager.mkdirs(str(path.parent))
    path.parent.mkdir(parents=True, exist_ok=True)

    # Older runs could leave a directory at `*.rrd` from the mkdirs(full_path) bug.
    if path.exists() and path.is_dir():
        shutil.rmtree(path)

    # with tempfile.NamedTemporaryFile(suffix=".rrd") as f:
    #     rerun.save(path=f.name)
    #     path_manager.copy(f.name, path, overwrite=True)

    rerun.save(str(path))


def destroy() -> None:
    rerun.disconnect()


def to_torch(*args) -> Tuple[torch.Tensor, ...]:
    return tuple(torch.from_numpy(x) if isinstance(x, np.ndarray) else x for x in args)


def to_np(*args) -> Tuple[np.ndarray, ...]:
    return tuple(x.cpu().numpy() if isinstance(x, torch.Tensor) else x for x in args)


def log_video(
    entity_name: str,
    rgb: Union[torch.Tensor, np.ndarray],
    intrinsics: Union[torch.Tensor, np.ndarray],
    extrinsics: Union[torch.Tensor, np.ndarray],
    depth: Union[torch.Tensor, np.ndarray],
    uv_space: bool = False,
    postfix: str = "",
    show_frame_number: bool = False,
    dense_source_mask: Optional[Union[torch.Tensor, np.ndarray]] = None,
    pcd_lighten: float = 0.0,
) -> None:
    """Log RGB video + RGB-colored depth point cloud + camera poses to rerun.

    ``pcd_lighten`` in [0, 1] mixes the RGBD backdrop colors toward white so
    overlaid 3D tracks read more clearly (0 = full RGB, 1 = pure white).
    """
    rgb, intrinsics, extrinsics, depth = to_np(rgb, intrinsics, extrinsics, depth)
    if dense_source_mask is not None:
        dense_source_mask = to_np(dense_source_mask)[0]
        if dense_source_mask.ndim == 4:
            dense_source_mask = dense_source_mask[:, 0]
        _, _, h_rgb, w_rgb = rgb.shape
        if dense_source_mask.shape[1:] != (h_rgb, w_rgb):
            import torch.nn.functional as F

            mask_torch = torch.from_numpy(dense_source_mask).float().unsqueeze(1)
            mask_torch = F.interpolate(mask_torch, size=(h_rgb, w_rgb), mode="nearest")
            dense_source_mask = mask_torch[:, 0].numpy()
        dense_source_mask = dense_source_mask.astype(bool)
    depth_torch, intrinsics_torch, extrinsics_torch = to_torch(
        depth, intrinsics, extrinsics
    )

    if uv_space:
        pcds_torch = batch_unproject_uv_space(depth_torch, intrinsics_torch)
    else:
        pcds_torch = batch_unproject(depth_torch, intrinsics_torch, extrinsics_torch)
    pcd = np.ascontiguousarray(pcds_torch.numpy())

    num_frames, _channels, height, width = rgb.shape
    assert _channels == 3, "RGB should have 3 channels"

    lighten = float(np.clip(pcd_lighten, 0.0, 1.0))

    def _lighten_pcd_colors(colors: np.ndarray) -> np.ndarray:
        """Mix RGB toward white, preserving input range for rerun.

        Rerun float colors must stay in [0, 1]; uint8 colors stay in [0, 255].
        Emitting float values in [0, 255] makes the cloud look like noise.
        """
        out = np.asarray(colors, dtype=np.float32)
        if out.size == 0:
            return out
        # Unit-range float (typical demo path): mix toward 1.0.
        if float(out.max()) <= 1.0 + 1e-6:
            out = out * (1.0 - lighten) + 1.0 * lighten
            return np.clip(out, 0.0, 1.0).astype(np.float32)
        # Byte-range: mix toward 255 and return uint8.
        out = out * (1.0 - lighten) + 255.0 * lighten
        return np.clip(out, 0.0, 255.0).astype(np.uint8)

    world_from_cam0 = np.linalg.inv(extrinsics[0])
    translation0 = world_from_cam0[:3, 3]
    rotation0 = R.from_matrix(world_from_cam0[:3, :3])
    rerun.log(f"/", rerun.ViewCoordinates.RIGHT_HAND_Y_DOWN, static=True)
    rerun.log(
        f"/{entity_name}",
        rerun.Transform3D(
            translation=translation0,
            rotation=rerun.Quaternion(xyzw=rotation0.as_quat()),
        ),
        static=True,
    )
    
    if False:
        positions_all = np.concatenate(
            [pcd[j].transpose(1, 2, 0).reshape(-1, 3) for j in range(num_frames)],
            axis=0,
        )
        colors_all = np.concatenate(
            [rgb[j].transpose(1, 2, 0).reshape(-1, 3) for j in range(num_frames)],
            axis=0,
        )

        # Voxelize per-frame to reduce point count for rerun.
        from ..datasets.utils.geometry import voxelization, merge_per_frame_p2v_labels
        from torch_scatter import scatter_mean

        positions_torch = torch.from_numpy(positions_all).float()
        colors_torch = torch.from_numpy(colors_all)

        pts_per_frame = height * width
        positions_per_frame = positions_torch.reshape(num_frames, pts_per_frame, 3)
        p2v = voxelization(positions_per_frame, voxel_size=0.005)
        p2v_global = merge_per_frame_p2v_labels(p2v, base_offset=0)

        positions_vox = scatter_mean(positions_torch, p2v_global, dim=0)
        colors_vox = scatter_mean(colors_torch, p2v_global, dim=0)

        if dense_source_mask is not None:
            mask_flat = torch.from_numpy(
                np.concatenate([dense_source_mask[j].ravel() for j in range(num_frames)])
            ).float()
            mask_vox = scatter_mean(mask_flat, p2v_global, dim=0)
            valid_vox = mask_vox > 0.5
            positions_vox = positions_vox[valid_vox]
            colors_vox = colors_vox[valid_vox]

        positions_all = positions_vox.numpy()
        colors_all = _lighten_pcd_colors(colors_vox.numpy())

        rerun.log(
            f"/{entity_name}/world/pcd_all_frames{postfix}",
            rerun.Points3D(positions_all, colors=colors_all),
            static=True,
        )
            
    
    for i in range(num_frames):
        rerun.set_time("frameid", sequence=i)
        frame_positions = pcd[i].transpose(1, 2, 0).reshape(-1, 3)
        frame_colors = rgb[i].transpose(1, 2, 0).reshape(-1, 3)
        if dense_source_mask is not None:
            frame_mask = dense_source_mask[i].ravel()
            frame_positions = frame_positions[frame_mask]
            frame_colors = frame_colors[frame_mask]
        frame_colors = _lighten_pcd_colors(frame_colors)
        rerun.log(
            f"/{entity_name}/world/pcd{postfix}",
            rerun.Points3D(frame_positions, colors=frame_colors),
        )
        world_from_cam = np.linalg.inv(extrinsics[i])
        translation = world_from_cam[:3, 3]
        rotation = R.from_matrix(world_from_cam[:3, :3])
        rerun.log(
            f"/{entity_name}/world/camera{postfix}",
            rerun.Transform3D(
                translation=translation,
                rotation=rerun.Quaternion(xyzw=rotation.as_quat()),
            ),
        )
        rerun.log(
            f"/{entity_name}/world/camera/image{postfix}",
            rerun.Pinhole(
                image_from_camera=intrinsics[i],
                resolution=[width, height],
                camera_xyz=rerun.ViewCoordinates.RDF,
            ),
        )
        image = np.ascontiguousarray(rgb[i].transpose(1, 2, 0))
        if show_frame_number:
            import cv2

            image = image.copy()
            font_scale = max(0.45, min(height, width) / 420.0)
            thickness = max(1, int(round(min(height, width) / 320.0)))
            org = (max(6, width // 80), max(18, height // 18))
            cv2.putText(
                image,
                f"{i}",
                org,
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (0, 0, 0),
                thickness,
                cv2.LINE_AA,
            )
        rerun.log(f"/{entity_name}/world/camera/image{postfix}", rerun.Image(image))


def log_trajectory(
    entity_name: str,
    track_name: str,
    intrinsics: Union[torch.Tensor, np.ndarray],
    extrinsics: Union[torch.Tensor, np.ndarray],
    trajs: Union[torch.Tensor, np.ndarray],
    visibs: Union[torch.Tensor, np.ndarray],
    valids: Union[torch.Tensor, np.ndarray],
    queries: Optional[Union[torch.Tensor, np.ndarray]] = None,
    track_len: int = 8,
    cmap_name: str = "rainbow",
) -> None:
    num_frames, num_points, _ = trajs.shape

    intrinsics_torch, extrinsics_torch, trajs_torch = to_torch(
        intrinsics, extrinsics, trajs
    )
    trajs_2d_torch = batch_project(
        trajs_torch.reshape(num_frames * num_points, 3),
        torch.repeat_interleave(intrinsics_torch, num_points, dim=0),
        torch.repeat_interleave(extrinsics_torch, num_points, dim=0),
    ).reshape(num_frames, num_points, 2)
    trajs_2d = trajs_2d_torch.cpu().numpy()
    intrinsics, extrinsics, trajs, visibs, valids = to_np(
        intrinsics, extrinsics, trajs, visibs, valids
    )

    if queries is not None:
        queries = to_np(queries)[0]

    assert visibs.dtype == np.bool_, "Visibility should be boolean"
    assert valids.dtype == np.bool_, "Validity should be boolean"

    cmap = matplotlib.colormaps[cmap_name]
    norm = matplotlib.colors.Normalize()  # type: ignore

    colors = cmap(norm(trajs[0, :, 1]))

    for i in range(num_frames):
        rerun.set_time("frameid", sequence=i)

        colors_rgba = colors.copy()  # alpha may not work..
        colors_rgba[~valids[i], -1] = 0.0
        # colors_rgba[~visibs[i], -1] *= 0.2
        colors_rgba[~visibs[i], :3] = colors_rgba[
            ~visibs[i], :3
        ] * 0.4 + 0.6 * np.array([0, 0, 0])

        rerun.log(
            f"/{entity_name}/world/{track_name}/points_3d",
            rerun.Points3D(
                trajs[i][valids[i]], colors=colors_rgba[valids[i]], radii=0.005
            ),
        )
        rerun.log(
            f"/{entity_name}/world/camera/image/{track_name}/points_2d",
            rerun.Points2D(
                trajs_2d[i][valids[i]], colors=colors_rgba[valids[i]], radii=0.005
            ),
        )

        if queries is not None:
            queries_at_i = queries[queries[..., 0] == i][..., 1:]
            rerun.log(
                f"/{entity_name}/world/{track_name}/queries_3d",
                rerun.Points3D(
                    queries_at_i,
                    colors=np.array([255, 255, 255], dtype=np.uint8),
                    radii=0.01,
                ),
            )

        if i >= 1:
            visible_track_mask = valids[i] & visibs[i] & valids[i - 1] & visibs[i - 1]
            valid_track_mask = valids[i] & valids[i - 1]
            colors_rgba = colors_rgba.copy()
            colors_rgba[:, -1] = 0.5
            rerun.log(
                f"/{entity_name}/world/camera/image/{track_name}/tracks_2d_{i}",
                rerun.LineStrips2D(
                    trajs_2d[i - 1 : i + 1, valid_track_mask].transpose(1, 0, 2),
                    colors=colors_rgba[valid_track_mask],
                ),
            )
            rerun.log(
                f"/{entity_name}/world/{track_name}/tracks_3d_{i}",
                rerun.LineStrips3D(
                    trajs[i - 1 : i + 1, valid_track_mask].transpose(1, 0, 2),
                    colors=colors_rgba[valid_track_mask],
                ),
            )

        if i >= track_len and track_len > 0:
            rerun.log(
                f"/{entity_name}/world/camera/image/{track_name}/tracks_2d_{i-track_len}",
                rerun.Clear(recursive=True),
            )
            rerun.log(
                f"/{entity_name}/world/{track_name}/tracks_3d_{i-track_len}",
                rerun.Clear(recursive=True),
            )
