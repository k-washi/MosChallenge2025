import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.stats
import torch
from lightning.pytorch import LightningModule
from transformers import get_cosine_schedule_with_warmup  # pyright: ignore[reportPrivateImportUsage]

from track3.core.config import Config
from track3.core.models.utils import get_loss, get_model

OUTPUT_SQ_DIM = 2


class MOSPredictorModule(LightningModule):
    """Base class for a MOS predictor module.

    Args:
    ----
        config (Config): Configuration object containing model parameters.
        model (nn.Module): The neural network model to be used for prediction.
        optimizer (torch.optim.Optimizer): Optimizer for the model.
        scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler for the model.

    """

    def __init__(
        self,
        config: Config,
    ) -> None:
        """Initialize the MOSPredictorModule."""
        super().__init__()
        self.c = config
        self.model = get_model(config=config.model, model_name=config.model.model_name)
        self.loss = get_loss(config=config.loss, loss_name=config.loss.loss_name)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass through the model."""
        pred = self.model(x, attention_mask=attention_mask)
        if pred.dim() == OUTPUT_SQ_DIM:
            pred = pred.squeeze(1)
        return pred

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Training step for the model.

        Args:
        ----
            batch (dict): Batch of data.
            batch_idx (int): Index of the batch.

        Returns:
        -------
            torch.Tensor: Loss value.

        """
        _ = batch_idx
        wavs, attention_mask, mos_score, _, _ = batch
        pred = self(wavs, attention_mask=attention_mask)

        loss = self.loss(pred, mos_score)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_validation_epoch_start(self) -> None:
        """Start of the validation epoch."""
        super().on_validation_epoch_start()
        self.dataset_mos_list = []  # (wav_fp, pred_mos_score, true_mos_score)

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Validate step for the model.

        Args:
        ----
            batch (dict): Batch of data.
            batch_idx (int): Index of the batch.

        Returns:
        -------
            torch.Tensor: Loss value.

        """
        _ = batch_idx
        wavs, attention_mask, mos_score, _, wav_fp_list = batch
        pred = self(wavs, attention_mask=attention_mask)
        loss = self.loss(pred, mos_score)
        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.c.ml.test_batch_size)

        pred = pred.detach().cpu()
        mos_score = mos_score.detach().cpu()
        for wav_fp, pred_mos, true_mos in zip(wav_fp_list, pred, mos_score, strict=False):
            self.dataset_mos_list.append((wav_fp, pred_mos.item(), true_mos.item()))

        return loss

    def on_validation_epoch_end(self) -> None:
        """End of the validation epoch."""
        super().on_validation_epoch_end()
        result_map = {}
        for wav_fp, pred_mos, true_mos in self.dataset_mos_list:
            # audio name example. xxx/dataset_name/wav/audio.wav
            dataset_name = Path(wav_fp).parent.parent.name
            audio_name = Path(wav_fp).name
            if dataset_name not in result_map:
                result_map[dataset_name] = {}
                result_map[dataset_name]["pred"] = []
                result_map[dataset_name]["true"] = []
                result_map[dataset_name]["data"] = []

            # normalizeをもとに戻す
            pred_mos = (
                pred_mos * (self.c.data.label_norm_max - self.c.data.label_norm_min)
                + (self.c.data.label_min + self.c.data.label_max) / 2.0
            )
            true_mos = (
                true_mos * (self.c.data.label_norm_max - self.c.data.label_norm_min)
                + (self.c.data.label_min + self.c.data.label_max) / 2.0
            )
            result_map[dataset_name]["pred"].append(pred_mos)
            result_map[dataset_name]["true"].append(true_mos)
            result_map[dataset_name]["data"].append((audio_name, pred_mos, true_mos))
        for dataset_name, result in result_map.items():
            pred_list = np.array(result["pred"])
            true_list = np.array(result["true"])
            l1 = np.mean(np.abs(true_list - pred_list))
            lcc = np.corrcoef(true_list, pred_list)[0][1]
            srcc = scipy.stats.spearmanr(true_list, pred_list)[0]
            ktau = scipy.stats.kendalltau(true_list, pred_list)[0]
            assert isinstance(srcc, float), f"srcc is not a float: {srcc}"
            assert isinstance(ktau, float), f"ktau is not a float: {ktau}"
            self.log(f"val/{dataset_name}/l1", l1.item(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log(f"val/{dataset_name}/lcc", lcc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log(f"val/{dataset_name}/srcc", srcc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log(f"val/{dataset_name}/ktau", ktau, on_step=False, on_epoch=True, prog_bar=True, logger=True)

            output_df = pd.DataFrame(result["data"], columns=["audio", "pred", "true"])  # pyright: ignore[reportArgumentType]
            output_fp = Path(self.c.path.val_save_dir) / f"{self.current_epoch:05d}" / f"{dataset_name}.csv"
            output_fp.parent.mkdir(parents=True, exist_ok=True)
            output_df.to_csv(output_fp, index=False)

    def configure_optimizers(self) -> Any:
        """Configure the optimizer and learning rate scheduler."""
        no_decay = ["bias", "LayerNorm.weight"]

        # https://tma15.github.io/blog/2021/09/17/deep-learningbert%E5%AD%A6%E7%BF%92%E6%99%82%E3%81%ABbias%E3%82%84layer-normalization%E3%82%92weight-decay%E3%81%97%E3%81%AA%E3%81%84%E7%90%86%E7%94%B1/#weight-decay%E3%81%AE%E5%AF%BE%E8%B1%A1%E5%A4%96%E3%81%A8%E3%81%AA%E3%82%8B%E3%83%91%E3%83%A9%E3%83%A1%E3%83%BC%E3%82%BF
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.c.ml.optimizer.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]

        if self.c.ml.optimizer.optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                optimizer_grouped_parameters,
                lr=self.c.ml.optimizer.lr,
                eps=self.c.ml.optimizer.adam_epsilon,
            )
        else:
            emsg = f"Optimizer {self.c.ml.optimizer.optimizer_name} not found."
            raise ValueError(emsg)
        # Scheduler
        assert self.c.data.train_dataset_num > 0, "train_dataset_num must be greater than 0"
        warmup_steps = int(
            math.floor(self.c.data.train_dataset_num / self.c.ml.batch_size / self.c.ml.accumulate_grad_num)
            * self.c.ml.optimizer.warmup_epoch
        )
        total_train_steps = int(
            math.floor(self.c.data.train_dataset_num / self.c.ml.batch_size / self.c.ml.accumulate_grad_num)
            * self.c.ml.num_epochs
        )
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_train_steps,
            num_cycles=self.c.ml.optimizer.num_cycles,
        )
        lr_scheduler_config = {
            "scheduler": scheduler,
            "interval": "step",
            "frequency": 1,
            "strict": True,
        }
        return [optimizer], [lr_scheduler_config]

    def lr_scheduler_step(self, scheduler: Any, metric: Any) -> None:
        """Step the learning rate scheduler.

        Args:
        ----
            scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
            metric (str): Metric to monitor.

        """
        _ = metric
        scheduler.step()
