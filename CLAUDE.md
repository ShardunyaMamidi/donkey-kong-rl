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
- `agent.py` — `EpsilonGreedy`, action-selection logic (exploration vs. a
  forward pass through the `DQN` network).
- `replay_buffer.py` — `ReplayBuffer`, experience replay storage/sampling.
- `train.py` — the actual training loop: env/network/target-network/optimizer
  setup, epsilon decay, gradient steps, target-network sync. `main.py` is
  still just wrapper-pipeline smoke tests; `train.py` is where real training
  happens.
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
- The target network itself (a second `DQN` instance) is built and managed in
  `train.py` (`make_target_network`), not `model.py` — see below.

## Agent (`agent.py`) — decisions made so far

- **`EpsilonGreedy`** — same exploration-vs-exploitation shape as the old
  tabular version (`frozen_lake_practice.py`), reworked for DQN:
  - Takes `rng` in the constructor (an `np.random.default_rng(...)` instance)
    instead of relying on a module-level global, so the class is
    self-contained/testable on its own.
  - Exploitation: adds a batch dim (`unsqueeze(0)`) to the single stacked
    observation, runs it through the network inside `torch.no_grad()` (no
    grad tracking needed for action selection), and takes a plain
    `argmax` — no tie-breaking logic like the old Q-table version had, since
    exact ties are effectively impossible with continuous network outputs.
  - Normalization is *not* done here — that lives inside `model.py`'s
    `forward()`, so `choose_action` just hands the raw uint8 observation to
    the network.
  - Takes a required `device` argument (no default) and moves the observation
    tensor there before the forward pass, since `net` may live on GPU.

## Replay buffer (`replay_buffer.py`) — decisions made so far

- **`ReplayBuffer`** stores `(state, action, reward, next_state, done)`
  transitions in a `deque(maxlen=capacity)` (same eviction pattern as
  `FrameSkip`/`FrameStack`).
- **Naive storage** — each transition stores its *full* `(4, 84, 84)` stacked
  state/next_state, not individual frames. This duplicates frame data across
  transitions (~4x memory vs. storing unique frames and reconstructing
  stacks), but is far simpler to get right first. Flagged as a known future
  optimization, not implemented yet.
- **`sample(batch_size)`** uses `random.sample` (without replacement) and
  returns plain **numpy arrays** (not torch tensors) — keeps this module
  framework-agnostic, same way `wrappers.py` has no torch dependency. The
  training loop (a later step) converts to tensors when needed.
- `done` matters for the loss computation: when `done` is true there's
  no valid `next_state` to bootstrap value from, so the target should not
  include the discounted next-state term for that sample.

## Training loop (`train.py`) — decisions made so far

- **`make_env()`** builds the full wrapper chain (same order as documented
  above) — the one place env construction happens for real training.
- **Device: CUDA if available, else CPU** (`torch.device(...)`); `net` and
  `target_net` are both moved to it, and `EpsilonGreedy` is given the same
  device so its action-selection tensor lands in the right place.
- **Target network (`make_target_network`)** — builds a second `DQN`,
  copies weights via `load_state_dict`, sets `.eval()`, and disables grad on
  all its params (`requires_grad_(False)`) since it's never trained directly,
  only synced.
- **Target sync is a hard periodic copy** (`todo.md`'s "periodically sync
  target network"), gated on `train_step` — a counter that only increments
  once a real gradient step has happened (i.e. only after buffer warm-up),
  not on raw environment steps. Verified correct via a one-off diagnostic
  script (mutate `net`'s weights, confirm divergence, sync, confirm they
  match again) — script was deleted after confirming, not kept in the repo.
- **Loss: Huber (`nn.SmoothL1Loss`)**, not MSE — more robust to outlier
  TD-errors than MSE, matches the original DQN paper. **Optimizer: Adam**,
  `learning_rate=1e-4`.
- **`gamma=0.99`** — deliberately not reusing the old FrozenLake value
  (`0.95`); Atari's longer reward horizons call for a higher discount factor.
- **Gradient step**: `predicted_q` via `net(states).gather(1, actions...)`
  (Q-value of the action actually taken); `target_q` via
  `rewards + gamma * target_net(next_states).max(...) * (1 - dones)` inside
  `torch.no_grad()`. Only `net`'s parameters are registered with the
  optimizer — `target_net` only ever changes via the periodic sync.
- **Epsilon decay (todo step 9): linear, per-step (not per-episode)** — from
  `epsilon_start` to `epsilon_end` over `epsilon_decay_steps` global steps,
  then held constant. Per-step chosen over per-episode since episode lengths
  vary, making step-based decay more predictable/reproducible.
- **Hyperparameters are currently untuned placeholders** (function defaults
  in `train()`) — `num_episodes`, `batch_size`, `buffer_capacity`,
  `warmup_steps`, `epsilon_*`, `target_sync_steps`, `gamma`,
  `learning_rate`. The `__main__` block runs with deliberately small values
  as a fast smoke test, not real training.

