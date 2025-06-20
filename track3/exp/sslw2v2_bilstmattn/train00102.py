"""Wav2vec2 linear training script.

dropout: 0.1 -> 0.3
lr: 2e-5 -> 5e-5
"""

import shutil
from pathlib import Path

import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

from track3.core.config import Config
from track3.core.dataset.base.mospl import MOSDataModule
from track3.core.models.sslpl import MOSPredictorModule
from track3.exp.utils import CheckpointEveryEpoch, get_datset_fp_label_mean

cfg = Config()
seed_everything(cfg.ml.seed)
##########
# Params #
##########

VERSION = "00102"
EXP_ID = "sslw2v2_bilstmattn"
WANDB_PROJECT_NAME = "moschallenge2025track3"
IS_LOGGING = True
FAST_DEV_RUN = False

LOG_SAVE_DIR = f"logs/{EXP_ID}/v{VERSION}"
Path(LOG_SAVE_DIR).mkdir(parents=True, exist_ok=True)
shutil.copy(__file__, LOG_SAVE_DIR)

user_csv_fp = "data/mos/user.csv"
TRAIN_LIST = [
    "data/mos/bvccmain/train.csv",
    "data/mos/bvccood/train.csv",
    "data/mos/somos/train.csv",
]
VAL_LIST = [
    "data/mos/bvccmain/val.csv",
    "data/mos/bvccood/val.csv",
    "data/mos/somos/val.csv",
    "data/mos/track3/utt_16k.csv",
    "data/mos/track3/utt_24k.csv",
    "data/mos/track3/utt_48k.csv",
]

TEST_LIST = [
    "data/mos/track3/utt_16k.csv",
    "data/mos/track3/utt_24k.csv",
    "data/mos/track3/utt_48k.csv",
]


train_audio_list, train_label_list = get_datset_fp_label_mean(
    dataset_list=TRAIN_LIST,
)
print(f"train_len: {len(train_audio_list)}")
val_audio_list, val_label_list = get_datset_fp_label_mean(
    dataset_list=VAL_LIST,
)
print(f"val_len: {len(val_audio_list)}")
test_audio_list, test_label_list = get_datset_fp_label_mean(
    dataset_list=TEST_LIST,
)

print(f"test_len: {len(test_audio_list)}")
##########
cfg.ml.num_epochs = 10
cfg.ml.batch_size = 16
cfg.ml.test_batch_size = 16
cfg.ml.num_workers = 4
cfg.ml.accumulate_grad_num = 1
cfg.ml.grad_clip_val = 1
cfg.ml.check_val_every_n_epoch = 1
cfg.ml.mix_precision = "32"

cfg.ml.optimizer.optimizer_name = "adamw"
cfg.ml.optimizer.lr = 5e-5
cfg.ml.optimizer.weight_decay = 0.005
cfg.ml.optimizer.adam_epsilon = 1e-5
cfg.ml.optimizer.warmup_epoch = 1
cfg.ml.optimizer.num_cycles = 0.5

cfg.path.model_save_dir = f"{LOG_SAVE_DIR}/ckpt"
cfg.path.val_save_dir = f"{LOG_SAVE_DIR}/val"

# model
cfg.model.model_name = "wav2vec2_bilstmattn"
cfg.model.w2v2.pretrained_model_name = "facebook/wav2vec2-base-960h"
cfg.model.w2v2.dropout = 0.3
cfg.model.w2v2.is_freeze_ssl = False

# dataset
cfg.data.pitch_shift_max = 2
cfg.data.time_wrap_max = 1.1
cfg.data.time_wrap_min = 0.9


def train() -> None:
    """Train the model."""
    dataset = MOSDataModule(
        config=cfg,
        train_audio_list=train_audio_list,
        train_label_list=train_label_list,
        val_audio_list=val_audio_list,
        val_label_list=val_label_list,
        test_audio_list=test_audio_list,
        test_label_list=test_label_list,
    )
    cfg.data.train_dataset_num = len(train_audio_list)
    model = MOSPredictorModule(cfg)
    logger = None
    if IS_LOGGING:
        logger = WandbLogger(
            project=WANDB_PROJECT_NAME,
            name=f"{EXP_ID}/v{VERSION}",
        )
    ckpt_callback = CheckpointEveryEpoch(save_dir=cfg.path.model_save_dir, every_n_epochs=cfg.ml.check_val_every_n_epoch)
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
        check_val_every_n_epoch=cfg.ml.check_val_every_n_epoch,
        logger=logger,
        callbacks=callback_list,
        num_sanity_val_steps=2,
    )
    trainer.fit(model, dataset)


if __name__ == "__main__":
    train()
