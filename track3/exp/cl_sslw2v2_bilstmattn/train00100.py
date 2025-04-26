"""Wav2vec2 linear training script."""

import shutil
from pathlib import Path

import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

from track3.core.config import Config
from track3.core.dataset.contrastive.mospl import MOSDataModule
from track3.core.dataset.contrastive.utils import get_labeldata_list, get_nolabel_list
from track3.core.models.sslclpl import MOSPredictorModule
from track3.exp.utils import CheckpointEverySteps

cfg = Config()
seed_everything(cfg.ml.seed)
##########
# Params #
##########

VERSION = "01104"
EXP_ID = "cl_sslw2v2_bilstmattn"
WANDB_PROJECT_NAME = "moschallenge2025track3_v2"
IS_LOGGING = True
FAST_DEV_RUN = False

LOG_SAVE_DIR = f"logs/{EXP_ID}/v{VERSION}"
Path(LOG_SAVE_DIR).mkdir(parents=True, exist_ok=True)
shutil.copy(__file__, LOG_SAVE_DIR)

# Contrastive learning用のラベルなしデータ
train_contrastive_list = []
NO_LABEL_TRAIN_LIST = ["/data/mosranking/libritts"]
for dataset_dir in NO_LABEL_TRAIN_LIST:
    tmp_train_nolabel_list = get_nolabel_list(dataset_dir=dataset_dir)
    train_contrastive_list.extend(tmp_train_nolabel_list)

TRAIN_LIST = [
    "/data/mosranking/bvccmain/train.csv",
    "/data/mosranking/somos/train.csv",
]
_train_contrastive_list, train_dataset_list = get_labeldata_list(
    dataset_csv_list=TRAIN_LIST,
)
train_contrastive_list.extend(_train_contrastive_list)
train_dataset_list = []
print(f"train_len: {len(train_dataset_list)}")
print(f"train_contrastive_len: {len(train_contrastive_list)}")

VAL_LIST = [
    "/data/mosranking/bvccmain/val.csv",
    "/data/mosranking/somos/val.csv",
    "/data/mosranking/track3/fold_0.csv",
    "/data/mosranking/track3/fold_1.csv",
    "/data/mosranking/track3/fold_2.csv",
    "/data/mosranking/track3/fold_3.csv",
    "/data/mosranking/track3/fold_4.csv",
]

val_contrastive_list, val_dataset_list = get_labeldata_list(
    dataset_csv_list=VAL_LIST,
)
val_contrastive_list = []

print(f"val_len: {len(val_dataset_list)}")
print(f"val_contrastive_len: {len(val_contrastive_list)}")

cfg.ml.num_epochs = 1
cfg.ml.batch_size = 6
cfg.ml.test_batch_size = 6
cfg.ml.num_workers = 4
cfg.ml.accumulate_grad_num = 4
cfg.ml.grad_clip_val = 1
cfg.ml.check_val_every_n_steps = 5000
cfg.ml.mix_precision = "32"

cfg.ml.optimizer.optimizer_name = "adamw"
cfg.ml.optimizer.lr = 8e-6
cfg.ml.optimizer.weight_decay = 0.01
cfg.ml.optimizer.adam_epsilon = 1e-8
cfg.ml.optimizer.warmup_epoch = 0.005  # 全エポックの1割くらい
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
cfg.data.time_wrap_max = 1.05
cfg.data.time_wrap_min = 0.95
cfg.data.is_label_normalize = True
cfg.data.is_extend = True
cfg.data.extend_rate = 0

# loss
cfg.loss.l1_rate_min = 0.0
cfg.loss.l1_rate_max = 0.0
cfg.loss.cl_rate = 0.0
cfg.loss.rank_rate = 100
cfg.loss.l1_loss_margin = 0.05
cfg.loss.contrastive_loss_margin = 0.2


def train() -> None:
    """Train the model."""
    dataset = MOSDataModule(
        config=cfg,
        train_dataset_list=train_dataset_list,
        train_contrastive_dataset_list=train_contrastive_list,
        val_dataset_list=val_dataset_list,
        val_contrastive_dataset_list=val_contrastive_list,
        test_dataset_list=[],
        test_contrastive_dataset_list=[],
    )
    cfg.data.train_dataset_num = len(dataset.train_dataset.dataset_list)
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
