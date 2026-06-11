"""Callback that stops training when the mean episode reward plateaus.

Used by the curriculum-training driver: each stage trains at a fixed
``v_max`` until either (a) the configured `stage_max_steps` is hit, or
(b) the mean episode reward over the last `window_size` samples has not
improved by at least `min_improve` reward units. Returning ``False`` from
``_on_step`` is SB3's idiom for early-stopping ``model.learn``.
"""

from __future__ import annotations

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class PlateauStopper(BaseCallback):
    """Stop training when episode reward plateaus, or max steps reached.

    Args:
        stage_max_steps: hard cap on this stage's training (≈ upper bound
            even if the policy keeps improving).
        stage_min_steps: minimum steps before plateau detection can fire.
            Prevents a very-early plateau decision when the warm-up window
            is still being populated.
        sample_every: take a mean-reward sample every this many steps.
        window_size: number of consecutive samples (no improvement) needed
            to declare a plateau. window_size * sample_every is the rough
            wall-clock window we require to be flat.
        min_improve: minimum improvement (max − start) across the window
            below which we call it plateaued.
        verbose: 0 silent, 1 print plateau / stage-end messages.
    """

    def __init__(self,
                 stage_max_steps: int,
                 stage_min_steps: int = 200_000,
                 sample_every: int = 10_000,
                 window_size: int = 20,
                 min_improve: float = 1.0,
                 verbose: int = 1):
        super().__init__(verbose)
        self.stage_max_steps = int(stage_max_steps)
        self.stage_min_steps = int(stage_min_steps)
        self.sample_every = int(sample_every)
        self.window_size = int(window_size)
        self.min_improve = float(min_improve)
        self._stage_start_step: int | None = None
        self._next_sample_step: int | None = None
        self._samples: list[tuple[int, float]] = []

    def _on_training_start(self) -> None:
        # Reset bookkeeping every time .learn() is called (i.e. each stage).
        self._stage_start_step = int(self.num_timesteps)
        self._next_sample_step = self._stage_start_step + self.sample_every
        self._samples = []

    def _on_step(self) -> bool:
        elapsed = self.num_timesteps - (self._stage_start_step or 0)

        if elapsed >= self.stage_max_steps:
            if self.verbose:
                print(f'[PlateauStopper] stage_max_steps ({self.stage_max_steps}) '
                      f'reached at elapsed={elapsed}')
            return False

        if self._next_sample_step is None or self.num_timesteps < self._next_sample_step:
            return True
        self._next_sample_step = self.num_timesteps + self.sample_every

        buf = self.model.ep_info_buffer
        if buf is None or len(buf) < 10:
            return True

        mean_r = float(np.mean([ep['r'] for ep in buf]))
        self._samples.append((int(self.num_timesteps), mean_r))
        if len(self._samples) > self.window_size:
            self._samples = self._samples[-self.window_size:]

        if elapsed < self.stage_min_steps:
            return True
        if len(self._samples) < self.window_size:
            return True

        values = [r for _, r in self._samples]
        start_val = values[0]
        max_val = max(values)
        improvement = max_val - start_val
        if improvement < self.min_improve:
            if self.verbose:
                print(f'[PlateauStopper] PLATEAU at step={self.num_timesteps} '
                      f'(elapsed={elapsed}): window {self.window_size} samples, '
                      f'max-start improvement {improvement:+.3f} '
                      f'< threshold {self.min_improve}')
            return False
        return True
