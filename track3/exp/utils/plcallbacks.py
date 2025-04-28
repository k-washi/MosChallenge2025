from pathlib import Path
from typing import Any

import lightning.pytorch as pl
import torch
from lightning.pytorch.utilities.types import STEP_OUTPUT


class CheckpointEveryEpoch(pl.Callback):
    """Checkpoint every n epochs."""

    def __init__(self, save_dir: str, start_epoch: int = 0, every_n_epochs: int = 1) -> None:
        """Checkpoint every n epochs."""
        self.start_epoch = start_epoch
        self.save_dir = save_dir
        self.epochs = 0
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Check if we should save a checkpoint after every train epoch."""
        _ = pl_module
        self.epochs += 1
        if self.epochs >= self.start_epoch and self.epochs % self.every_n_epochs == 0:
            save_dir = Path(f"{self.save_dir}") / f"ckpt-{self.epochs}"
            save_dir.mkdir(exist_ok=True, parents=True)
            save_path = save_dir / "model.ckpt"

            if hasattr(trainer, "model"):
                if hasattr(trainer.model, "module"):
                    torch.save(trainer.model.module.state_dict(), save_path)  # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue]

                else:
                    torch.save(trainer.model.state_dict(), save_path)  # pyright: ignore[reportOptionalMemberAccess]
                return

            emsg = "Model not found."
            raise ValueError(emsg)


class CheckpointEverySteps(pl.Callback):
    """Checkpoint every n step."""

    def __init__(self, save_dir: str, every_n_steps: int = 10000) -> None:
        """Checkpoint every n epochs."""
        self.save_dir = save_dir
        self.every_n_steps = every_n_steps

    def on_train_batch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: STEP_OUTPUT, batch: Any, batch_idx: int
    ) -> None:
        """Check if we should save a checkpoint after every train epoch."""
        _ = pl_module
        _, _, _ = outputs, batch, batch_idx
        global_step = trainer.global_step * trainer.accumulate_grad_batches
        if global_step % self.every_n_steps == 0 and global_step > 0:
            save_dir = Path(f"{self.save_dir}") / f"ckpt-{global_step}"
            save_dir.mkdir(exist_ok=True, parents=True)
            save_path = save_dir / "model.ckpt"
            if hasattr(trainer, "model"):
                if hasattr(trainer.model, "module"):
                    torch.save(trainer.model.module.state_dict(), save_path)  # pyright: ignore[reportOptionalMemberAccess, reportAttributeAccessIssue]

                else:
                    torch.save(trainer.model.state_dict(), save_path)  # pyright: ignore[reportOptionalMemberAccess]
                return

            emsg = "Model not found."
            raise ValueError(emsg)
