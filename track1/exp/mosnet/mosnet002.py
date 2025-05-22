"""Wav2vec2 linear training script.

track3を訓練に追加し,fold likeにする
"""

import shutil
from pathlib import Path

import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

from track1.core.config import Config
from track1.core.dataset.getdataset import get_dataset
from track1.core.dataset.mospl_aux import MOSDataModule
from track1.core.model.mospl_aux import MOSPredictorModule
from track3.exp.utils import CheckpointEverySteps

cfg = Config()
seed_everything(cfg.ml.seed)
##########
# Params #
##########

VERSION = "01005"
EXP_ID = "mosnet_aux"
WANDB_PROJECT_NAME = "moschallenge2025track1_v1"
IS_LOGGING = True
FAST_DEV_RUN = False

LOG_SAVE_DIR = f"logs/{EXP_ID}/v{VERSION}"
Path(LOG_SAVE_DIR).mkdir(parents=True, exist_ok=True)
shutil.copy(__file__, LOG_SAVE_DIR)

# データの読み込み

train_audio_list, train_prompt_list, train_score_list = get_dataset(
    dataset_dir="data/MusicEval-phase1",
    mos_text="data/MusicEval-phase1/sets/train_mos_list.txt",
)

print(f"train_len: {len(train_audio_list)}")

val_audio_list, val_prompt_list, val_score_list = get_dataset(
    dataset_dir="data/MusicEval-phase1",
    mos_text="data/MusicEval-phase1/sets/dev_mos_list.txt",
)

print(f"val_len: {len(val_audio_list)}")

cfg.ml.num_epochs = 30
cfg.ml.batch_size = 10
cfg.ml.test_batch_size = 10
cfg.ml.num_workers = 4
cfg.ml.accumulate_grad_num = 1
cfg.ml.grad_clip_val = 1.0
cfg.ml.check_val_every_n_steps = 192
cfg.ml.mix_precision = "32"

cfg.ml.optimizer.lr = 5e-5
cfg.ml.optimizer.weight_decay = 0.001

cfg.path.model_save_dir = f"{LOG_SAVE_DIR}/ckpt"
cfg.path.val_save_dir = f"{LOG_SAVE_DIR}/val"

# model
cfg.model.model_name = "mosnet_aux"
cfg.model.mosnet.input_dim = 512
cfg.model.mosnet.aux_dim = 557
cfg.model.mosnet.hidden_dim = 1024
cfg.model.mosnet.tf_n_head = 16
cfg.model.mosnet.tf_n_layers = 3
cfg.model.mosnet.tf_dropout = 0.15
cfg.model.mosnet.head_dropout = 0.3

# dataset
cfg.data.aug_rate = 0.3

# loss
cfg.loss.l1_rate_min = 0.5
cfg.loss.l1_rate_max = 1
cfg.loss.cl_rate = 0.5  # 順序にはこっちがきく
cfg.loss.diff_rate = 0.5  # l1が安定
cfg.loss.l1_loss_margin = 0.1
cfg.loss.contrastive_loss_margin = 0.1


def train() -> None:
    """Train the model."""
    dataset = MOSDataModule(
        config=cfg,
        train_audio_list=train_audio_list,
        train_prompt_list=train_prompt_list,
        train_score_list=train_score_list,
        val_audio_list=val_audio_list,
        val_prompt_list=val_prompt_list,
        val_score_list=val_score_list,
    )
    cfg.data.train_dataset_num = len(dataset.train_dataset.audio_list)
    model = MOSPredictorModule(cfg)
    logger = None
    if IS_LOGGING:
        logger = WandbLogger(
            project=WANDB_PROJECT_NAME,
            name=f"{EXP_ID}/v{VERSION}",
        )
    ckpt_callback = CheckpointEverySteps(
        save_dir=cfg.path.model_save_dir,
        every_n_steps=cfg.ml.check_val_every_n_steps,
    )
    callback_list = [
        ckpt_callback,
        LearningRateMonitor(logging_interval="step"),
    ]
    ################################
    # 訓練
    ################################
    device = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = Trainer(
        precision=cfg.ml.mix_precision,  # pyright: ignore[reportArgumentType]
        accelerator=device,
        devices="auto",
        max_epochs=cfg.ml.num_epochs,
        accumulate_grad_batches=cfg.ml.accumulate_grad_num,
        fast_dev_run=FAST_DEV_RUN,
        gradient_clip_val=cfg.ml.grad_clip_val,
        val_check_interval=cfg.ml.check_val_every_n_steps,
        check_val_every_n_epoch=None,
        logger=logger,
        callbacks=callback_list,
        num_sanity_val_steps=2,
    )
    trainer.fit(model, dataset)


if __name__ == "__main__":
    train()
