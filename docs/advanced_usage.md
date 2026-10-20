# Advanced Usage Guide

This guide covers advanced inference modes, scene representation parameters, memory optimization in TrackEverything.

---

## 1. Scene Voxelization Strategies

TrackEverything de-duplicates 3D points into a voxel grid to maintain a persistent scene memory. You can configure how the scene space is discretized:

### Uniform Voxelization (Default)
Standard Cartesian grid where all voxels have fixed side length:
```bash
python -m trackeverything.demo --video path/to/video.mp4 --voxel_mode uniform --voxel_size 0.02
```

### Signed-Log Voxelization (`signedlog`)
Logarithmic binning along each axis, calibrated to have grid edge equal to `voxel_size` at unit distance from the camera:
```bash
python -m trackeverything.demo --video path/to/video.mp4 --voxel_mode signedlog --voxel_size 0.02
```
*When to use:* I recommend trying this for expansive outdoor scenes or drone footage where depth varies across multiple orders of magnitude. It provides high precision for objects near the camera while keeping distant background points from flooding the token budget. Note that this is experimental but I have found this to work well for my use cases. This, however, adds a lot more points to track if you keep the voxel size to 0.02 still. I recommend trying larger voxel sizes and see if that works for you.

### Endpoint Voxelization (`--voxelize_on_endpoints`)
After the initial iteration, points whose predicted 3D endpoint positions fall into the same voxel are merged. This accelerates refinement iterations without degrading accuracy. Enabled by default; disable with `--voxelize_on_endpoints false` if strict token identity retention is needed. I detail this in the supplementary of our paper.

---

## 2. Long-Horizon Memory Compression

In videos spanning thousands of frames, even static voxel memory can grow large. To place an upper bound on token count, pass `--compress_memory N`:

```bash
python -m trackeverything.demo \
  --video path/to/long_video.mp4 \
  --compress_memory 8000
```

When the static token count exceeds `N`, the memory is automatically coarsened by merging adjacent voxels, allowing tracking to continue indefinitely in constant GPU memory. This is experimental too and not well tested. I recommend not using this unless you are sure you need it.

---
