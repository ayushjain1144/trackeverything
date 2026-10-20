# Frequently Asked Questions (FAQ)

A collection of common questions, troubleshooting tips, and practical details for working with TrackEverything.

---

### Inference & Memory


#### 1. Can I track only specific points instead of every pixel?

Not really. The closest you can get is to track those specific points in addition to every pixel. This is done by supplying query points to the model. On top of this, you can also use the `--mode sparse` flag which does not keep around the points from previous window while tracking -- this can be significantly cheap and fast for some videos where the scene memory grows long. And in my experiments, it doesn't seem to affect the accuracy of the query point tracking much.

#### 2. I am still running into memory issues on long videos, what should I do?

Play around with `--voxel_size` to see if a larger voxel size suffices for your use case. Next, you can try using the `--compress_memory` flag to compress the memory: this will merge static points only with higher voxel size if they exceed a certain number of points. This is an experimental feature and not well tested. Another thing to try is subsample the number of frames in the video (i.e. go lower frame rate).

---

#### 3. Does the tracker always need 16 frames, or can it run online?

The tracker is trained with 16-frame windows, but it can work reasonably well down to a window size of 4 frames (you can change this by setting `model.seq_len=4`).

If you really want to track every frame as it comes in, setting the window size directly to 1 does not work — advancing and re-voxelizing memory every single frame causes the memory representation to quickly collapse, hurting tracking accuracy significantly.

What has worked in practice is to maintain an accumulating buffer up to the window size (e.g. 16 frames):

- **Frame 1 arrives:** we track on frame 1.
- **Frame 2 arrives:** we track across [frame 1, frame 2].
- ...
- **Frame 14 arrives:** we add it to the buffer [frames 1–13] and track across all 14 frames. We can initialize the trajectories of the first 13 frames from the prior prediction to help the model converge faster.
- **Buffer fills (frame 16):** once the number of frames reaches the buffer limit, we commit the state and clear the buffer.
- For the next frame, the buffer resets and the new window starts with just that 1 frame.

Note that this buffering strategy is very experimental. In our tests, it did not hurt tracking performance, but running multiple passes over growing frame buffers is noticeably slower than tracking with fixed windows of 4, 8, or 16 frames. I would recommend sticking to fixed window sizes (or testing window sizes of 4/8/16) unless you want to experiment with online streaming on your own.

---

---
