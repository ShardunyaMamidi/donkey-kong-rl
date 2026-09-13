# DQN for DonkeyKong - Roadmap

Porting the tabular Q-learning approach (`frozen_lake_practice.py`) to a DQN for
`ALE/DonkeyKong-v5`. Raw observations are images, so a Q-table won't work —
this tracks the steps to move from table lookups to a neural network.

## Steps

- [x] **1. Preprocessing pipeline** - grayscale, resize/downsample (e.g. 84x84),
      normalize pixel values, stack last N frames (e.g. 4) for motion info
- [x] **2. Environment wrappers** - frame-skipping w/ max-pooling over last 2
      frames, reward clipping (-1/0/1), optional episodic-life handling
      (check `AtariPreprocessing` / frame-stack wrappers in gymnasium/ale-py)
- [x] **3. Replace Q-table with a neural network (CNN)** - takes stacked frames
      as input, outputs one Q-value per action (replaces `self.qtable`)
- [x] **4. Rework `choose_action`** - forward pass through network instead of
      `q_table[state, :]`; keep epsilon-greedy logic
- [x] **5. Add a replay buffer (experience replay)** - store
      `(state, action, reward, next_state, done)` transitions, train on random
      mini-batches instead of updating every step
- [x] **6. Add a target network** - slowly-updated copy of the network used to
      compute target Q-values, for training stability
- [x] **7. Define loss + optimizer** - MSE/Huber loss between predicted and
      target Q, backprop via Adam (replaces the tabular update rule)
- [x] **8. Restructure training loop** - reset -> act -> step -> push to buffer
      -> periodically sample batch + gradient step -> periodically sync target
      network
- [x] **9. Epsilon decay schedule** - decay from ~1.0 to ~0.01-0.1 over
      training instead of a fixed epsilon
- [x] **10. New hyperparameters** - replay buffer size, batch size, target
      network update frequency, learning rate (~1e-4), frame-stack size,
      buffer warm-up period
- [x] **11. Pick deep learning framework** - PyTorch or TensorFlow (numpy
      alone can't do backprop through a CNN)
- [ ] **12. Rework plotting/postprocessing** - drop the Q-table heatmap
      (no discrete state grid anymore); use reward-per-episode curves, loss
      curves, and/or rendered gameplay video instead

## Notes / Decisions

- (log framework choice, hyperparameter values, and any deviations here as
  they're decided)
