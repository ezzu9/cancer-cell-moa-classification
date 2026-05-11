"""Training callbacks: early stopping and LR warm-up scheduler."""

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def get_warmup_scheduler(optimiser: Optimizer, warmup_steps: int) -> LambdaLR:
    """Linear warm-up over `warmup_steps` steps, then constant."""
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 1.0
    return LambdaLR(optimiser, lr_lambda)


class EarlyStopping:
    def __init__(self, patience: int = 10, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best = -float("inf")
        self.counter = 0

    def step(self, val_metric: float) -> bool:
        """Returns True if training should stop."""
        if val_metric > self.best + self.min_delta:
            self.best = val_metric
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience
