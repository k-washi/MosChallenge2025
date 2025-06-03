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

from track1.core.config import Config
from track1.core.model.getmodel import get_model
from track3.core.models.utils import mosdiff_loss, ranknet_loss

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
        self.model = get_model(config=config)
        self.total_train_steps = int(
            math.floor(self.c.data.train_dataset_num / self.c.ml.batch_size / self.c.ml.accumulate_grad_num)
            * self.c.ml.num_epochs
        )

    def forward(
        self,
        z_audio: torch.Tensor,
        z_prompt: torch.Tensor,
        z_aux: torch.Tensor,
        wav: torch.Tensor,
        wav_len: torch.Tensor,
        text_ids: torch.Tensor,
        text_len: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the model."""
        pred_mi, pred_ta = self.model(
            z_audio,
            z_prompt,
            z_aux,
            wav,
            wav_len,
            text_ids,
            text_len,
        )
        if pred_mi.dim() == OUTPUT_SQ_DIM:
            pred_mi = pred_mi.squeeze(1)
        if pred_ta.dim() == OUTPUT_SQ_DIM:
            pred_ta = pred_ta.squeeze(1)
        return pred_mi, pred_ta

    def _caculate_loss(
        self, pred_mi: torch.Tensor, pred_ta: torch.Tensor, score_mi: torch.Tensor, score_ta: torch.Tensor
    ) -> tuple:
        """Calculate the loss for the model.

        Args:
        ----
            pred_mi (torch.Tensor): Predicted MOS scores for MI.
            pred_ta (torch.Tensor): Predicted MOS scores for TA.
            score_mi (torch.Tensor): True MOS scores for MI.
            score_ta (torch.Tensor): True MOS scores for TA.

        Returns:
        -------
            return losses

        """
        # マージンありL1lossの計算
        # miに関して
        mask = (score_mi >= self.c.data.label_norm_min).float()
        diff1 = torch.abs(pred_mi - score_mi).clamp(max=2.0)
        l1loss_1 = F.smooth_l1_loss(
            diff1, torch.zeros_like(diff1, device=pred_mi.device), reduction="none", beta=self.c.loss.l1_loss_margin
        )
        l1loss_1 = torch.sum(mask * l1loss_1) / (mask.sum() + 1e-6)

        mask = (score_ta >= self.c.data.label_norm_min).float()
        diff2 = torch.abs(pred_ta - score_ta).clamp(max=2.0)
        l1loss_2 = F.smooth_l1_loss(
            diff2, torch.zeros_like(diff2, device=pred_ta.device), reduction="none", beta=self.c.loss.l1_loss_margin
        )
        l1loss_2 = torch.sum(mask * l1loss_2) / (mask.sum() + 1e-6)

        rank_loss_mi = ranknet_loss(pred_mi, score_mi, self.c.loss.contrastive_loss_margin)
        rank_loss_ta = ranknet_loss(pred_ta, score_ta, self.c.loss.contrastive_loss_margin)

        diff_loss_mi = mosdiff_loss(pred_mi, score_mi, self.c.loss.contrastive_loss_margin)
        diff_loss_ta = mosdiff_loss(pred_ta, score_ta, self.c.loss.contrastive_loss_margin)
        return l1loss_1, l1loss_2, rank_loss_mi, rank_loss_ta, diff_loss_mi, diff_loss_ta

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
        z_audio, z_prompt, z_aux, score_mi, score_ta, _, wav, wav_m, text_ids, text_m = batch
        pred_mi, pred_ta = self(
            z_audio,
            z_prompt,
            z_aux,
            wav,
            wav_m,
            text_ids,
            text_m,
        )

        l1loss_mi, l1loss_ta, rank_loss_mi, rank_loss_ta, diff_loss_mi, diff_loss_ta = self._caculate_loss(
            pred_mi, pred_ta, score_mi, score_ta
        )
        l1_rate = self.c.loss.l1_rate_min + (self.c.loss.l1_rate_max - self.c.loss.l1_rate_min) * (
            self.global_step / self.total_train_steps
        )
        loss = (
            l1_rate * (l1loss_mi + l1loss_ta)
            + self.c.loss.rank_rate * (rank_loss_mi + rank_loss_ta)
            + self.c.loss.diff_rate * (diff_loss_mi + diff_loss_ta)
        )

        self.log("train/l1_mi", l1loss_mi, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train/l1_ta", l1loss_ta, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train/rank_mi", rank_loss_mi, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train/rank_ta", rank_loss_ta, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train/diff_mi", diff_loss_mi, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train/diff_ta", diff_loss_ta, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_validation_epoch_start(self) -> None:
        """Start of the validation epoch."""
        super().on_validation_epoch_start()
        self.dataset_mos_mi_list = []  # (wav_fp, pred_mos_score, true_mos_score)
        self.dataset_mos_ta_list = []  # (wav_fp, pred_mos_score, true_mos_score)
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
        z_audio, z_prompt, z_aux, score_mi, score_ta, wav_fp_list, wav, wav_m, text_ids, text_m = batch
        pred_mi, pred_ta = self(
            z_audio,
            z_prompt,
            z_aux,
            wav,
            wav_m,
            text_ids,
            text_m,
        )

        l1loss_mi, l1loss_ta, rank_loss_mi, rank_loss_ta, diff_loss_mi, diff_loss_ta = self._caculate_loss(
            pred_mi, pred_ta, score_mi, score_ta
        )
        l1_rate = self.c.loss.l1_rate_min + (self.c.loss.l1_rate_max - self.c.loss.l1_rate_min) * (
            self.global_step / self.total_train_steps
        )
        loss = (
            l1_rate * (l1loss_mi + l1loss_ta)
            + self.c.loss.rank_rate * (rank_loss_mi + rank_loss_ta)
            + self.c.loss.diff_rate * (diff_loss_mi + diff_loss_ta)
        )
        self.log("val/l1_mi", l1loss_mi, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val/l1_ta", l1loss_ta, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val/rank_mi", rank_loss_mi, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val/rank_ta", rank_loss_ta, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val/diff_mi", diff_loss_mi, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val/diff_ta", diff_loss_ta, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, batch_size=self.c.ml.test_batch_size)

        pred_mi = pred_mi.detach().cpu()
        pred_ta = pred_ta.detach().cpu()

        score_mi = score_mi.detach().cpu()
        score_ta = score_ta.detach().cpu()

        for wav_fp, p_mi, true_mi, p_ta, true_ta in zip(wav_fp_list, pred_mi, score_mi, pred_ta, score_ta, strict=False):
            self.dataset_mos_mi_list.append((wav_fp, p_mi.item(), true_mi.item()))
            self.dataset_mos_ta_list.append((wav_fp, p_ta.item(), true_ta.item()))

        return loss

    def on_validation_epoch_end(self) -> None:
        """End of the validation epoch."""
        super().on_validation_epoch_end()
        result_map = {}
        for (wav_fp_mi, pred_mos_mi, true_mos_mi), (_, pred_mos_ta, true_mos_ta) in zip(
            self.dataset_mos_mi_list, self.dataset_mos_ta_list, strict=False
        ):
            # audio name example. xxx/dataset_name/wav/audio.wav
            dataset_name = Path(wav_fp_mi).parent.parent.name
            audio_name = Path(wav_fp_mi).name
            if dataset_name not in result_map:
                result_map[dataset_name] = {}
                result_map[dataset_name]["pred_mi"] = []
                result_map[dataset_name]["true_mi"] = []
                result_map[dataset_name]["pred_ta"] = []
                result_map[dataset_name]["true_ta"] = []
                result_map[dataset_name]["data"] = []

            # normalizeをもとに戻す
            pred_mos_mi = (
                pred_mos_mi * (self.c.data.label_norm_max - self.c.data.label_norm_min)
                + (self.c.data.label_min + self.c.data.label_max) / 2.0
            )
            true_mos_mi = (
                true_mos_mi * (self.c.data.label_norm_max - self.c.data.label_norm_min)
                + (self.c.data.label_min + self.c.data.label_max) / 2.0
            )

            pred_mos_ta = (
                pred_mos_ta * (self.c.data.label_norm_max - self.c.data.label_norm_min)
                + (self.c.data.label_min + self.c.data.label_max) / 2.0
            )
            true_mos_ta = (
                true_mos_ta * (self.c.data.label_norm_max - self.c.data.label_norm_min)
                + (self.c.data.label_min + self.c.data.label_max) / 2.0
            )

            result_map[dataset_name]["pred_mi"].append(pred_mos_mi)
            result_map[dataset_name]["true_mi"].append(true_mos_mi)
            result_map[dataset_name]["pred_ta"].append(pred_mos_ta)
            result_map[dataset_name]["true_ta"].append(true_mos_ta)
            result_map[dataset_name]["data"].append((audio_name, pred_mos_mi, pred_mos_ta, true_mos_mi, true_mos_ta))

        for dataset_name, result in result_map.items():
            pred_list = np.array(result["pred_mi"])
            true_list = np.array(result["true_mi"])
            l1 = np.mean(np.abs(true_list - pred_list))
            lcc = np.corrcoef(true_list, pred_list)[0][1]
            srcc = scipy.stats.spearmanr(true_list, pred_list)[0]
            ktau = scipy.stats.kendalltau(true_list, pred_list)[0]
            assert isinstance(srcc, float), f"srcc is not a float: {srcc}"
            assert isinstance(ktau, float), f"ktau is not a float: {ktau}"
            self.log(f"val/{dataset_name}/mi/l1", l1.item(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log(f"val/{dataset_name}/mi/lcc", lcc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log(f"val/{dataset_name}/mi/srcc", srcc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log(f"val/{dataset_name}/mi/ktau", ktau, on_step=False, on_epoch=True, prog_bar=True, logger=True)

            pred_list = np.array(result["pred_ta"])
            true_list = np.array(result["true_ta"])
            l1 = np.mean(np.abs(true_list - pred_list))
            lcc = np.corrcoef(true_list, pred_list)[0][1]
            srcc = scipy.stats.spearmanr(true_list, pred_list)[0]
            ktau = scipy.stats.kendalltau(true_list, pred_list)[0]
            assert isinstance(srcc, float), f"srcc is not a float: {srcc}"
            assert isinstance(ktau, float), f"ktau is not a float: {ktau}"
            self.log(f"val/{dataset_name}/ta/l1", l1.item(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log(f"val/{dataset_name}/ta/lcc", lcc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log(f"val/{dataset_name}/ta/srcc", srcc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log(f"val/{dataset_name}/ta/ktau", ktau, on_step=False, on_epoch=True, prog_bar=True, logger=True)

            output_df = pd.DataFrame(result["data"], columns=["audio", "pred_mi", "pred_ta", "true_mi", "true_ta"])  # pyright: ignore[reportArgumentType]
            output_fp = Path(self.c.path.val_save_dir) / f"{self.global_step}" / f"{dataset_name}.csv"
            output_fp.parent.mkdir(parents=True, exist_ok=True)
            output_df.to_csv(output_fp, index=False)
        self.optimizers().train()  # pyright: ignore[reportAttributeAccessIssue]

    def configure_optimizers(self) -> Any:
        """Configure the optimizer and learning rate scheduler."""
        no_decay = ["bias", "LayerNorm.weight"]

        # https://tma15.github.io/blog/2021/09/17/deep-learningbert%E5%AD%A6%E7%BF%92%E6%99%82%E3%81%ABbias%E3%82%84layer-normalization%E3%82%92weight-decay%E3%81%97%E3%81%AA%E3%81%84%E7%90%86%E7%94%B1/#weight-decay%E3%81%AE%E5%AF%BE%E8%B1%A1%E5%A4%96%E3%81%A8%E3%81%AA%E3%82%8B%E3%83%91%E3%83%A9%E3%83%A1%E3%83%BC%E3%82%BF

        grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],  # pyright: ignore[reportAttributeAccessIssue]
                "weight_decay": self.c.ml.optimizer.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],  # pyright: ignore[reportAttributeAccessIssue]
                "weight_decay": 0.0,
            },
        ]

        optimizer = RAdamScheduleFree(
            grouped_parameters,
            lr=self.c.ml.optimizer.lr,
        )
        return [optimizer], []
