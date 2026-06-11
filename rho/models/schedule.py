import abc
import math
from dataclasses import asdict, dataclass

import torch
from draccus import ChoiceRegistry
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ConstantLR, LambdaLR, LRScheduler


@dataclass
class LRSchedulerConfig(ChoiceRegistry, abc.ABC):
    @property
    def type(self) -> str:
        return self.get_choice_name(self.__class__)

    @abc.abstractmethod
    def build(self, optimizer: Optimizer, num_training_steps: int) -> LRScheduler | None:
        raise NotImplementedError


@LRSchedulerConfig.register_subclass("constant")
@dataclass
class ConstantLRConfig(LRSchedulerConfig):
    name: str = "constant"

    def build(self, optimizer: torch.optim.Optimizer, num_training_steps: int) -> LRScheduler:
        return ConstantLR(optimizer, factor=1.0, total_iters=1)


@LRSchedulerConfig.register_subclass("diffuser")
@dataclass
class DiffuserSchedulerConfig(LRSchedulerConfig):
    num_warmup_steps: int
    name: str = "cosine"

    def build(self, optimizer: torch.optim.Optimizer, num_training_steps: int) -> LambdaLR:
        from diffusers.optimization import get_scheduler

        kwargs = {**asdict(self), "num_training_steps": num_training_steps, "optimizer": optimizer}
        return get_scheduler(**kwargs)


@LRSchedulerConfig.register_subclass("cosine_decay_with_warmup")
@dataclass
class CosineDecayWithWarmupSchedulerConfig(LRSchedulerConfig):
    """Used by Physical Intelligence to train Pi0"""

    num_warmup_steps: int
    num_decay_steps: int
    peak_lr: float
    decay_lr: float
    decay_start_step: int | None = (
        None  # If set, decay starts after this step instead of immediately after warmup
    )

    name: str = "cosine_decay_with_warmup"

    def build(self, optimizer: Optimizer, num_training_steps: int) -> LRScheduler:
        del num_training_steps

        # If decay_start_step is not set, decay starts immediately after warmup.
        # Clamp so decay never begins before warmup finishes.
        decay_start = self.decay_start_step if self.decay_start_step is not None else self.num_warmup_steps
        decay_start = max(decay_start, self.num_warmup_steps)

        def lr_lambda(current_step):
            def linear_warmup_schedule(current_step):
                if current_step <= 0:
                    return 1 / (self.num_warmup_steps + 1)
                frac = 1 - current_step / self.num_warmup_steps
                return (1 / (self.num_warmup_steps + 1) - 1) * frac + 1

            def cosine_decay_schedule(current_step):
                # Offset so the cosine starts at 1.0 when decay begins
                step = min(current_step - decay_start, self.num_decay_steps)
                cosine_decay = 0.5 * (1 + math.cos(math.pi * step / self.num_decay_steps))
                alpha = self.decay_lr / self.peak_lr
                decayed = (1 - alpha) * cosine_decay + alpha
                return decayed

            if current_step < self.num_warmup_steps:
                return linear_warmup_schedule(current_step)

            if current_step >= decay_start:
                return cosine_decay_schedule(current_step)

            return 1.0  # Keep peak LR until decay starts

        return LambdaLR(optimizer, lr_lambda, -1)


@LRSchedulerConfig.register_subclass("warmup_stable_decay")
@dataclass
class WSDSchedulerConfig(LRSchedulerConfig):
    """
    Warmup-Stable-Decay schedule with optional cycling (MiniCPM-style).

    Each cycle consists of:
      1. Warmup:  linear ramp from decay_lr to peak_lr
      2. Stable:  hold at peak_lr
      3. Decay:   cosine anneal from peak_lr to decay_lr

    The first cycle uses num_warmup_steps for warmup. Subsequent cycles
    (if num_cycles > 1) re-warm over num_rewarm_steps.

    Reference: "MiniCPM: Unveiling the Potential of Small Language Models
               with Scalable Training Strategies" (Hu et al., 2024)
    """

    num_warmup_steps: int
    num_decay_steps: int
    peak_lr: float
    decay_lr: float
    num_cycles: int = 1  # 1 = single WSD, >1 = repeated WSD cycles
    num_rewarm_steps: int | None = None  # Warmup steps for cycles 2+; defaults to num_warmup_steps

    name: str = "warmup_stable_decay"

    def build(self, optimizer: Optimizer, num_training_steps: int) -> LRScheduler:
        rewarm = self.num_rewarm_steps if self.num_rewarm_steps is not None else self.num_warmup_steps
        alpha = self.decay_lr / self.peak_lr  # floor ratio

        # Compute cycle boundaries
        # First cycle: warmup + stable + decay
        # Subsequent cycles: rewarm + stable + decay
        # Stable phase fills whatever steps remain after warmup+decay within each cycle
        if self.num_cycles <= 1:
            # Single cycle: warmup, then stable until decay_start, then decay
            cycle_starts = [0]
            total = num_training_steps
        else:
            first_cycle_min = self.num_warmup_steps + self.num_decay_steps
            later_cycle_min = rewarm + self.num_decay_steps
            remaining = num_training_steps - first_cycle_min
            if self.num_cycles > 1 and remaining > 0:
                later_total = remaining
                steps_per_later_cycle = max(later_total // (self.num_cycles - 1), later_cycle_min)
            else:
                steps_per_later_cycle = later_cycle_min

            cycle_starts = [0]
            for i in range(1, self.num_cycles):
                if i == 1:
                    cycle_starts.append(first_cycle_min + (steps_per_later_cycle - later_cycle_min))
                    # Adjust: first cycle gets its share of stable phase
                    cycle_starts[1] = num_training_steps - (self.num_cycles - 1) * steps_per_later_cycle
                    cycle_starts[1] = max(cycle_starts[1], first_cycle_min)
                else:
                    cycle_starts.append(cycle_starts[-1] + steps_per_later_cycle)
            total = num_training_steps

        def lr_lambda(current_step):
            # Determine which cycle we're in
            cycle_idx = 0
            for i in range(len(cycle_starts) - 1, -1, -1):
                if current_step >= cycle_starts[i]:
                    cycle_idx = i
                    break

            cycle_start = cycle_starts[cycle_idx]
            cycle_end = cycle_starts[cycle_idx + 1] if cycle_idx + 1 < len(cycle_starts) else total
            step_in_cycle = current_step - cycle_start
            cycle_length = cycle_end - cycle_start

            warmup_steps = self.num_warmup_steps if cycle_idx == 0 else rewarm
            stable_steps = max(cycle_length - warmup_steps - self.num_decay_steps, 0)
            decay_start_in_cycle = warmup_steps + stable_steps

            if step_in_cycle < warmup_steps:
                # Linear warmup from alpha to 1.0
                if warmup_steps == 0:
                    return 1.0
                t = step_in_cycle / warmup_steps
                return alpha + (1.0 - alpha) * t
            elif step_in_cycle < decay_start_in_cycle:
                # Stable phase at peak
                return 1.0
            else:
                # Cosine decay from 1.0 to alpha
                decay_step = step_in_cycle - decay_start_in_cycle
                decay_step = min(decay_step, self.num_decay_steps)
                cosine_decay = 0.5 * (1 + math.cos(math.pi * decay_step / self.num_decay_steps))
                return (1.0 - alpha) * cosine_decay + alpha

        return LambdaLR(optimizer, lr_lambda, -1)
