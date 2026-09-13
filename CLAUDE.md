# donkey-kong-rl

Porting a tabular Q-learning agent (`frozen_lake_practice.py`, FrozenLake) to a
DQN that plays `ALE/DonkeyKong-v5`. Raw observations are images, so this
project is about building the pipeline a CNN needs (preprocessing, replay
buffer, target network, etc.) instead of a Q-table.

The authoritative roadmap and step-by-step status lives in `todo.md` — check
it for what's done and what's next before assuming project state from memory.

## File layout

- `main.py` — env setup + a `__main__` sanity-check block that builds the
  wrapper chain and prints observation shapes/dtypes after `reset()`/`step()`.
  Not a real test suite — just a manual smoke test.
- `wrappers.py` — all `gym.Wrapper` subclasses (preprocessing pipeline).
- `model.py` — the CNN (`DQN` class) that replaces the Q-table.
- `frozen_lake_practice.py` — old tabular Q-learning reference implementation.
  Not part of the DonkeyKong pipeline; kept for comparison only.
- `todo.md` — roadmap tracking DQN porting steps.

## Environment

- `ALE/DonkeyKong-v5` via `gym.make(..., obs_type="grayscale")` — grayscale
  conversion happens in ALE itself, not manually in Python.
- Run scripts with `.venv/bin/python main.py` (or `uv run main.py`). Bare
  `python`/`python3` on PATH is not this project's interpreter and won't have
  `gymnasium`/`ale_py`/`cv2` installed.

## Preprocessing pipeline (`wrappers.py`) — decisions made so far

Wrapper chain order (each wraps the previous):

```
raw env -> FrameSkip -> ClipReward -> ResizeObservation -> FrameStack
```

- **`FrameSkip`** (`gym.Wrapper`, `skip=4`): repeats one action for `skip` raw
  emulator steps, sums reward across them, and returns the pixel-wise max of
  the last 2 raw frames (handles Atari's flickering-sprite rendering quirk).
  Breaks early on `terminated`/`truncated`.
- **`ClipReward`** (`gym.RewardWrapper`): `np.sign(reward)` -> -1/0/1. Placed
  directly after `FrameSkip` so it clips the already-summed skip-window
  reward, not each sub-frame's reward individually.
- **`ResizeObservation`** (`gym.ObservationWrapper`): plain `cv2.resize` to
  84x84 with `INTER_AREA` (no cropping — revisit only if training struggles
  and the HUD/score region turns out to be noise worth removing).
- **`FrameStack`** (`gym.ObservationWrapper`, `num_frames=4`): keeps a
  `deque(maxlen=4)`; `reset()` is overridden explicitly to fill the deque with
  4 copies of the first frame (the default `ObservationWrapper.reset()` flow
  would only append one frame and not clear stale frames from a prior
  episode); `step()` uses the inherited `observation()` hook to append+stack.

Other decisions:
- **Normalization is deferred to batch/training time** — frames stay `uint8`
  all the way through preprocessing and (eventually) the replay buffer, to
  keep memory down. Division by 255 happens right before the network forward
  pass, not in `wrappers.py`.
- **Episodic-life handling is intentionally not implemented yet** — deferred
  until/unless training stability needs it (see `todo.md` step 2).

## Network (`model.py`) — decisions made so far

- **Framework: PyTorch.** Chosen over TensorFlow for more idiomatic/eager
  debugging and closer alignment with common DQN reference implementations.
- **`DQN` class** — the classic "Nature DQN" architecture, sized for the
  `(4, 84, 84)` uint8 input `FrameStack` produces:
  - Conv1: 32 filters, 8x8, stride 4 -> Conv2: 64 filters, 4x4, stride 2 ->
    Conv3: 64 filters, 3x3, stride 1 (ReLU after each)
  - Flatten (`64 * 7 * 7 = 3136`, hardcoded — verified via a dummy forward
    pass rather than computed dynamically) -> FC 512 -> FC `num_actions` (18
    for Donkey Kong), no output activation (raw Q-values).
- **Normalization lives inside `forward()`** (`x.float() / 255.0` as the
  first op), not in `wrappers.py` or the future replay buffer. Rationale:
  keeps the model self-contained so every caller (training, and later
  inference/`choose_action`) automatically gets correctly-scaled input
  without having to remember a separate normalization step.
- No target network yet — that's a distinct later step (`todo.md` step 6);
  `model.py` currently defines only the one network architecture.

