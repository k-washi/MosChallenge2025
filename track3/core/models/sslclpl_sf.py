import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.stats
import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from schedulefree import RAdamScheduleFree

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
        self.total_train_steps = int(
            math.floor(self.c.data.train_dataset_num / self.c.ml.batch_size / self.c.ml.accumulate_grad_num)
            * self.c.ml.num_epochs
        )

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Forward pass through the model."""
        pred = self.model(x, attention_mask=attention_mask)
        if pred.dim() == OUTPUT_SQ_DIM:
            pred = pred.squeeze(1)
        return pred

    def _caculate_loss(
        self, pred1: torch.Tensor, pred2: torch.Tensor, mos_score1: torch.Tensor, mos_score2: torch.Tensor
    ) -> tuple:
        """Calculate the loss for the model.

        Args:
        ----
            pred1 (torch.Tensor): Predictions from the model.
            pred2 (torch.Tensor): Predictions from the model.
            mos_score1 (torch.Tensor): True MOS scores for the first set of predictions.
            mos_score2 (torch.Tensor): True MOS scores for the second set of predictions.

        Returns:
        -------
            return losses

        """
        # マージンありL1lossの計算
        mask = (mos_score1 >= self.c.data.label_norm_min).float()
        diff1 = torch.abs(pred1 - mos_score1).clamp(max=2.0)
        l1loss_1 = F.smooth_l1_loss(
            diff1, torch.zeros_like(diff1, device=pred1.device), reduction="none", beta=self.c.loss.l1_loss_margin
        )
        l1loss_1 = torch.sum(mask * l1loss_1) / (mask.sum() + 1e-6)

        mask = (mos_score2 >= self.c.data.label_norm_min).float()
        diff2 = torch.abs(pred2 - mos_score2).clamp(max=2.0)
        l1loss_2 = F.smooth_l1_loss(
            diff2, torch.zeros_like(diff2, device=pred2.device), reduction="none", beta=self.c.loss.l1_loss_margin
        )
        l1loss_2 = torch.sum(mask * l1loss_2) / (mask.sum() + 1e-6)

        # MOSPO
        # mosの情報がない場合id1 > id2となる
        mask = (
            (mos_score1 < self.c.data.label_norm_min)
            | (mos_score2 < self.c.data.label_norm_min)
            | (
                (mos_score1 >= self.c.data.label_norm_min)
                & (mos_score2 >= self.c.data.label_norm_min)
                & (mos_score1 > mos_score2)
            )
        ).float()
        loss_r = torch.sum(
            mask
            * F.margin_ranking_loss(
                input1=pred1.clamp(-1, 1),
                input2=pred2.clamp(-1, 1),
                target=torch.ones_like(pred1, dtype=torch.float32, device=pred1.device),
                margin=self.c.loss.ranking_loss_margin,
                reduction="none",
            )
        ) / (mask.sum() + 1e-6)

        # UTMOS
        # mosの情報がある場合id1 > id2はlossとして活用し id1 == id2の場合は無視する
        mask = ((mos_score1 >= self.c.data.label_norm_min) & (mos_score2 >= self.c.data.label_norm_min)).float()
        true_mos_diff = (mos_score1 - mos_score2).clamp(-2.0, 2.0)
        pred_mos_diff = (pred1 - pred2).clamp(-2.0, 2.0)
        loss_c = torch.sum(mask * F.relu(torch.abs(true_mos_diff - pred_mos_diff) - self.c.loss.contrastive_loss_margin)) / (
            mask.sum() + 1e-6
        )
        return l1loss_1, l1loss_2, loss_r, loss_c

    def on_fit_start(self) -> None:
        """Start of the fit process."""
        super().on_fit_start()
        self.optimizers().train()  # pyright: ignore[reportAttributeAccessIssue]

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
        wavs1, attention_mask1, mos_score1, _, wavs2, attention_mask2, mos_score2, _ = batch
        pred = self(torch.cat([wavs1, wavs2], dim=0), attention_mask=torch.cat([attention_mask1, attention_mask2], dim=0))
        pred1, pred2 = torch.split(pred, [len(wavs1), len(wavs2)], dim=0)

        l1loss_1, l1loss_2, loss_r, loss_c = self._caculate_loss(pred1, pred2, mos_score1, mos_score2)
        l1_rate = self.c.loss.l1_rate_min + (self.c.loss.l1_rate_max - self.c.loss.l1_rate_min) * (
            self.global_step / self.total_train_steps
        )
        loss = l1_rate * (l1loss_1 + l1loss_2) + self.c.loss.rank_rate * loss_r + self.c.loss.cl_rate * loss_c

        self.log("train/l1loss01", l1loss_1, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train/l1loss02", l1loss_2, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train/loss_r", loss_r, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train/loss_c", loss_c, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_validation_epoch_start(self) -> None:
        """Start of the validation epoch."""
        super().on_validation_epoch_start()
        self.dataset_mos_list = []  # (wav_fp, pred_mos_score, true_mos_score)
        self.optimizers().eval()  # pyright: ignore[reportAttributeAccessIssue]

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
        wavs1, attention_mask1, mos_score1, wav_fp_list, wavs2, attention_mask2, mos_score2, _ = batch
        pred = self(torch.cat([wavs1, wavs2], dim=0), attention_mask=torch.cat([attention_mask1, attention_mask2], dim=0))
        pred1, pred2 = torch.split(pred, [len(wavs1), len(wavs2)], dim=0)

        l1loss_1, l1loss_2, loss_r, loss_c = self._caculate_loss(pred1, pred2, mos_score1, mos_score2)
        l1_rate = self.c.loss.l1_rate_min + (self.c.loss.l1_rate_max - self.c.loss.l1_rate_min) * (
            self.global_step / self.total_train_steps
        )
        loss = l1_rate * (l1loss_1 + l1loss_2) + self.c.loss.rank_rate * loss_r + self.c.loss.cl_rate * loss_c
        self.log("val/l1loss01", l1loss_1, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val/l1loss02", l1loss_2, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val/loss_r", loss_r, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val/loss_c", loss_c, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.c.ml.test_batch_size)

        mask = mos_score1 >= self.c.data.label_norm_min
        pred1 = pred1[mask].detach().cpu()
        mos_score1 = mos_score1[mask].detach().cpu()
        wav_fp_list = [wav_fp for i, wav_fp in enumerate(wav_fp_list) if mask[i]]
        for wav_fp, pred_mos, true_mos in zip(wav_fp_list, pred1, mos_score1, strict=False):
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
            output_fp = Path(self.c.path.val_save_dir) / f"{self.global_step}" / f"{dataset_name}.csv"
            output_fp.parent.mkdir(parents=True, exist_ok=True)
            output_df.to_csv(output_fp, index=False)
        self.optimizers().train()  # pyright: ignore[reportAttributeAccessIssue]

    def configure_optimizers(self) -> Any:
        """Configure the optimizer and learning rate scheduler."""
        no_decay = ["bias", "LayerNorm.weight"]

        # https://tma15.github.io/blog/2021/09/17/deep-learningbert%E5%AD%A6%E7%BF%92%E6%99%82%E3%81%ABbias%E3%82%84layer-normalization%E3%82%92weight-decay%E3%81%97%E3%81%AA%E3%81%84%E7%90%86%E7%94%B1/#weight-decay%E3%81%AE%E5%AF%BE%E8%B1%A1%E5%A4%96%E3%81%A8%E3%81%AA%E3%82%8B%E3%83%91%E3%83%A9%E3%83%A1%E3%83%BC%E3%82%BF

        ssl_grouped_parameters = [
            {
                "params": [p for n, p in self.model.ssl_model.named_parameters() if not any(nd in n for nd in no_decay)],  # pyright: ignore[reportAttributeAccessIssue]
                "weight_decay": self.c.ml.optimizer.weight_decay,
                "lr": self.c.ml.optimizer.ssl_lr,
            },
            {
                "params": [p for n, p in self.model.ssl_model.named_parameters() if any(nd in n for nd in no_decay)],  # pyright: ignore[reportAttributeAccessIssue]
                "weight_decay": 0.0,
                "lr": self.c.ml.optimizer.ssl_lr,
            },
        ]
        head_optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.head.named_parameters() if not any(nd in n for nd in no_decay)],  # pyright: ignore[reportAttributeAccessIssue]
                "weight_decay": self.c.ml.optimizer.weight_decay,
                "lr": self.c.ml.optimizer.head_lr,
            },
            {
                "params": [p for n, p in self.model.head.named_parameters() if any(nd in n for nd in no_decay)],  # pyright: ignore[reportAttributeAccessIssue]
                "weight_decay": 0.0,
                "lr": self.c.ml.optimizer.head_lr,
            },
        ]

        optimizer = RAdamScheduleFree(
            [*ssl_grouped_parameters, *head_optimizer_grouped_parameters],
        )
        return [optimizer], []
