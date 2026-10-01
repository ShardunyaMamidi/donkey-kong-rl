# donkey-kong-rl

Started as a port of a tabular Q-learning agent (`frozen_lake_practice.py`,
FrozenLake) to a DQN that plays `ALE/DonkeyKong-v5`. Now restructured as a
small multi-algorithm framework: a `BaseAgent` interface + factory, with DQN
as the first implementation and PPO (and DQN variants) planned next.

The authoritative roadmap and step-by-step status lives in `todo.md` — check
it for what's done and what's next before assuming project state from memory.

## File layout

- `main.py` — entry point for a real training run: `--algo` (default `dqn`,
  choices come from the factory registry) -> `train(algo, make_config(algo))`
  using the config's default values. **Running `main.py` starts a real,
  possibly long-running training job**, not a smoke test.
- `config.py` — `BaseConfig`, frozen `kw_only` dataclass of settings every
  algorithm shares (`total_episodes`, `seed`, `num_frames`, `frame_skip`,
  `checkpoint_dir`, `checkpoint_every`).
- `envs.py` — `make_env(cfg, render_mode=None)`, the one place the env +
  wrapper chain is built (used by both training and `play.py`).
- `wrappers.py` — all `gym.Wrapper` subclasses (preprocessing pipeline).
- `networks/encoder.py` — `NatureCNN`, the shared conv trunk (outputs 3136
  features); every algorithm's network puts its own head on top of it.
- `networks/q_network.py` — `QNetwork` = `NatureCNN` + FC Q-value head (DQN).
- `buffers/replay_buffer.py` — `ReplayBuffer`, experience replay
  storage/sampling (a PPO rollout buffer would live alongside it).
- `agents/base.py` — `BaseAgent` ABC: the hook interface the trainer drives.
- `agents/dqn.py` — `DQNConfig(BaseConfig)` + `DQNAgent(BaseAgent)`: network,
  target network, replay buffer, optimizer, epsilon decay, gradient step.
- `agents/exploration.py` — `EpsilonGreedy` and `linear_epsilon`.
- `factory.py` — `AGENTS` registry (name -> agent class), `make_config(algo,
  **overrides)`, `make_agent(algo, cfg, env, device)`.
- `trainer.py` — the algorithm-agnostic loop `train(algo, cfg)`: env
  stepping, episode bookkeeping, logging, periodic + final checkpoints. Its
  own `__main__` block runs a tiny DQN config as a fast smoke test (writes to
  `checkpoints/smoke/`).
- `play.py` — loads a checkpoint, rebuilds the matching config/env/agent from
  the algo + config stored inside it, and runs `agent.act(obs,
  explore=False)` with `render_mode="human"`. CLI args: `--checkpoint`
  (default `checkpoints/dqn_final.pt`), `--episodes` (default 5).
- `checkpoints/` — `{algo}_episode_{n}.pt` every `checkpoint_every` episodes
  plus `{algo}_final.pt`. Gitignored — large binary artifacts, not source.
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

## Agent architecture (`agents/base.py`, `factory.py`, `trainer.py`)

- **`BaseAgent` (ABC)** is the contract between an algorithm and the shared
  trainer. Per env step the trainer calls `act(obs)` ->
  `observe(obs, action, reward, next_obs, terminated, truncated)` ->
  `update()`. Abstract: `act(obs, explore=True)`, `observe`, `update`,
  `checkpoint_modules()` (name -> `nn.Module` to save). Optional:
  `log_stats()` (dict printed per episode, e.g. DQN's epsilon).
- **Why hooks + one shared loop (not each algo owning its loop)**: env
  stepping, logging and checkpointing are identical across algorithms. The
  off-policy/on-policy difference is absorbed by `update()`: DQN takes a
  gradient step every call once warmed up; PPO is meant to cache log-prob/
  value in `act()`, append to a rollout in `observe()`, and no-op `update()`
  until the rollout is full.
- **`observe` receives `terminated` and `truncated` separately** so
  algorithms that need the distinction (PPO/GAE bootstrapping) have it.
  `DQNAgent` currently collapses them to `done = terminated or truncated`
  (preserves pre-refactor behavior; strictly, truncation shouldn't cut off
  bootstrapping — a known, minor inaccuracy, unaddressed).
- **Factory (`factory.py`)**: explicit `AGENTS` dict, not a register
  decorator — adding an algorithm = implement `BaseAgent` + one import + one
  dict entry. Each agent class carries `name` and `config_cls` as
  `ClassVar`s, so `make_config(algo)` / `make_agent(algo, ...)` need only the
  name string.
- **Config: one frozen `kw_only` dataclass per algorithm**, subclassing
  `BaseConfig` (`DQNConfig` lives in `agents/dqn.py` next to its algorithm).
  Defaults *are* the real-training values; smoke tests/experiments override
  via `make_config(algo, field=value, ...)`. Replaced the old
  `Params(NamedTuple)` in `main.py` (NamedTuples can't be subclassed per
  algorithm), which also removed the old `main.py` <-> `train.py` circular
  import.
- **Algorithm variants (Double DQN, Dueling, PER, ...) are planned as config
  flags on `DQNConfig`**, not separate factory entries, so they compose
  freely. None implemented yet.
- **Checkpoint format**: `agent.checkpoint()` -> `{"algo", "config"
  (asdict), "modules": {name: state_dict}}`, saved with `torch.save`. Storing
  algo + config lets `play.py` rebuild the right agent from the file alone.
  Only weights needed for playback are saved (DQN: `net`, not `target_net`,
  optimizer or buffer) — enough to watch a trained agent, not to resume
  training exactly. Loads fine under `torch.load`'s default
  `weights_only=True` (plain dict/str/number/tensor content only).
- **Device: CUDA if available, else CPU**, chosen in `trainer.train()` /
  `play.py` and passed into the agent, which moves its own networks.

## Networks (`networks/`) — decisions made so far

- **Framework: PyTorch.** Chosen over TensorFlow for more idiomatic/eager
  debugging and closer alignment with common DQN reference implementations.
- **Split into shared encoder + per-algorithm head** so DQN and future PPO
  (actor-critic head) reuse the same conv trunk.
- **`NatureCNN` (`encoder.py`)** — the classic "Nature DQN" conv stack, sized
  for the `(4, 84, 84)` uint8 input `FrameStack` produces:
  - Conv1: 32 filters, 8x8, stride 4 -> Conv2: 64 filters, 4x4, stride 2 ->
    Conv3: 64 filters, 3x3, stride 1 (ReLU after each)
  - Flatten to `out_features = 64 * 7 * 7 = 3136` (hardcoded — verified via a
    dummy forward pass rather than computed dynamically).
- **`QNetwork` (`q_network.py`)** — `NatureCNN` -> FC 512 -> FC `num_actions`
  (18 for Donkey Kong), no output activation (raw Q-values).
- **Normalization lives inside `NatureCNN.forward()`** (`x.float() / 255.0`
  as the first op), not in `wrappers.py` or the replay buffer. Rationale:
  every network built on the encoder automatically gets correctly-scaled
  input, and callers hand it raw uint8 observations.

## DQN (`agents/dqn.py`, `agents/exploration.py`) — decisions made so far

- **`EpsilonGreedy`** — same exploration-vs-exploitation shape as the old
  tabular version (`frozen_lake_practice.py`), reworked for DQN:
  - Takes `rng` in the constructor (`np.random.default_rng(cfg.seed)`)
    instead of relying on a module-level global.
  - `greedy_action()`: adds a batch dim (`unsqueeze(0)`), runs the network
    inside `torch.no_grad()`, takes a plain `argmax` — no tie-breaking logic
    like the Q-table version, since exact ties are effectively impossible
    with continuous outputs. `choose_action()` uses it for the exploit
    branch; `DQNAgent.act(explore=False)` calls it directly (playback).
  - Takes a required `device` and moves the observation tensor there.
- **Target network (`make_target_network`)** — `copy.deepcopy(net)`, sets
  `.eval()`, disables grad on all params; never trained directly, only
  synced. (deepcopy rather than re-constructing, so it works unchanged for
  future network variants like dueling.)
- **Target sync is a hard periodic copy**, gated on `train_steps` — a counter
  that only increments once a real gradient step has happened (i.e. after
  buffer warm-up), not on raw env steps. Verified correct (pre-refactor) via
  a one-off diagnostic script, deleted after confirming.
- **Loss: Huber (`nn.SmoothL1Loss`)**, not MSE — more robust to outlier
  TD-errors, matches the original DQN paper. **Optimizer: Adam.**
- **`gamma=0.99`** — deliberately not the old FrozenLake `0.95`; Atari's
  longer reward horizons call for a higher discount factor.
- **Gradient step**: `predicted_q` via `net(states).gather(1, actions...)`;
  `target_q` via `rewards + gamma * target_net(next_states).max(...) *
  (1 - dones)` inside `torch.no_grad()`. Only `net`'s parameters are
  registered with the optimizer.
- **Epsilon decay: linear, per-step (not per-episode)** — `epsilon_start` ->
  `epsilon_end` over `epsilon_decay_steps` env steps (`DQNAgent.env_steps`),
  then held. Per-step since episode lengths vary.
- **Replay buffer (`buffers/replay_buffer.py`)**: `(state, action, reward,
  next_state, done)` in a `deque(maxlen=capacity)`. **Naive storage** — full
  `(4, 84, 84)` stacks per transition (~4x memory vs. storing unique frames);
  known future optimization. `sample()` uses `random.sample` and returns
  **numpy arrays** (framework-agnostic); the agent converts to tensors.

## Hyperparameters (`DQNConfig` defaults) — decisions made so far

- **Values are deliberately modest**, not the original DQN paper's scale
  (millions of frames, 1M-transition buffer) — chosen to keep a single-GPU
  hobby run reasonable: `total_episodes=500`, `buffer_capacity=20_000`,
  `epsilon_decay_steps=50_000`, `warmup_steps=1_000`,
  `target_sync_steps=1_000`, `batch_size=32`, `learning_rate=1e-4`,
  `gamma=0.99`. A starting point, not tuned values.
- **Known limitation (discussed, not yet acted on):** Donkey Kong is a
  historically hard game for vanilla DQN (sparse/delayed rewards, precise
  timing-dependent actions) — plain epsilon-greedy at this scale is unlikely
  to finish a level. Planned directions: DQN variants as `DQNConfig` flags
  (Double, Dueling, PER) and PPO as a second `BaseAgent` — none implemented
  yet.
