# Multiview Inference

We add an experimental version of zero-shot multiview inference with TrackEverything. Given multiview time synchronized cameras as input, this version of TrackEverything can track all points across all frames (and timesteps) of the video.

For multi-view inference, frames from multiple synchronized cameras are lifted into the shared world-coordinate feature cloud. By assigning all camera views captured at the identical timestamp the same source-time temporal positional encoding, the architecture naturally handles multi-camera setups zero-shot without requiring multi-view-specific heads, retraining, or special camera groupings.

We do not train on multiview data, and just do zero-shot inference on the multiview data. In our experiments, we saw the tracking performance to be similar or marginally worse with multiple views, compared to single view inference. Perhaps, training with multiview data will help yield performance boost.

---

## Running it

### On a set of videos

Pass one video per camera, separated by commas:

```bash
python -m trackeverything.demo_multiview \
  --model checkpoints/model.pth \
  --video cam0.mp4,cam1.mp4,cam2.mp4,cam3.mp4 \
  --save_dense_2d_video
```

The cameras must be time synchronized, and the videos must have the same
resolution. 

### From Python

```python
from trackeverything.demo_hf_multiview import MultiViewTracker

tracker = MultiViewTracker.from_pretrained("checkpoints/model.pth")
output = tracker.infer(video=["cam0.mp4", "cam1.mp4", "cam2.mp4", "cam3.mp4"])

tracks = output.tracks()   # one track per scene point
```

To save the tracks without making any videos, use the command line tool:

```bash
python -m trackeverything.demo_hf_multiview \
  --video cam0.mp4,cam1.mp4 --output_dir outputs/
```

### With your own depth and camera poses

The frames of all the cameras are placed on a single timeline, one timestep at a
time: timestep 0 camera 0, timestep 0 camera 1, ..., then timestep 1 camera 0,
and so on. So the position of a frame on that timeline is

```
position = timestep * number_of_cameras + camera
```

Your depth maps, intrinsics and extrinsics are read in that same order, so put
them in that order before passing them in:

```bash
python -m trackeverything.demo_multiview \
  --video cam0.mp4,cam1.mp4 \
  --depths_dir path/to/depths_folder \
  --intrinsics_path path/to/intrinsics.npy \
  --extrinsics_path path/to/extrinsics.npy
```

The same order applies to query points: a point you want to track on camera `v`
at timestep `f` is given the frame index `f * number_of_cameras + v`.

### Evaluation

Two multiview benchmarks are set up, Panoptic Studio and DexYCB. Point
`panoptic_multiview_root` and `dexycb_multiview_root` in
[`configs/paths.yaml`](../configs/paths.yaml) at your copies of the data, then:

```bash
CKPT=checkpoints/model.pth DATASET=panoptic VIEWS=4 FRAMES=16 \
  bash scripts/eval_multiview.sh
```

`DATASET` is `panoptic` or `dexycb`, and `VIEWS` is how many cameras to use. The
script picks the first `VIEWS` cameras of each scene and reports the metrics on
the first camera's frames, so the numbers can be compared across camera counts.

---

## Good to know

- **Inference only.** 
- Currently, we use only one camera for doing WAFT (feature correlation); using multiple cameras was worse for performance. However, this can be an artifact of zero-shot inference, so if you try training with multi-view data, I think it is worthwhile to try doing WAFT sampling to all cameras. This should especially help occlusions. 
