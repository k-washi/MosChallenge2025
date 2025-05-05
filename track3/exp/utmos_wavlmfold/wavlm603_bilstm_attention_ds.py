"""Wav2vec2 linear training script.

track3を訓練に追加し,fold likeにする
val:2,test:3
"""

import shutil
from pathlib import Path

import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

from track3.core.config import Config
from track3.core.dataset.contrastive.utils import get_labeldata_list
from track3.core.dataset.utmosds.mospl import MOSDataModule
from track3.core.models.utmospl_sfds import MOSPredictorModule
from track3.exp.utils import CheckpointEverySteps

cfg = Config()
seed_everything(cfg.ml.seed)
##########
# Params #
##########

VERSION = "03003"
EXP_ID = "utmos_sslwavlm_sfds_fold"
WANDB_PROJECT_NAME = "moschallenge2025track3_v2"
IS_LOGGING = True
FAST_DEV_RUN = False

LOG_SAVE_DIR = f"logs/{EXP_ID}/v{VERSION}"
Path(LOG_SAVE_DIR).mkdir(parents=True, exist_ok=True)
shutil.copy(__file__, LOG_SAVE_DIR)

# データの読み込み
TRAIN_LIST = [
    "/data/mosranking/bvccmain/train.csv",
    "/data/mosranking/somos/train.csv",
    "/data/mosranking/track3/fold_0.csv",
    "/data/mosranking/track3/fold_1.csv",
    "/data/mosranking/track3/fold_4.csv",
    "/data/mosranking/track3/fold_5.csv",
    "/data/mosranking/track3/fold_6.csv",
    "/data/mosranking/track3/fold_7.csv",
    "/data/mosranking/track3/fold_8.csv",
    "/data/mosranking/track3/fold_9.csv",
]
_, train_dataset_list = get_labeldata_list(dataset_csv_list=TRAIN_LIST, is_balanced=True)
print(f"train_len: {len(train_dataset_list)}")

VAL_LIST = [
    "/data/mosranking/bvccmain/val.csv",
    "/data/mosranking/somos/val.csv",
    "/data/mosranking/track3/fold_2.csv",
]

_, val_dataset_list = get_labeldata_list(
    dataset_csv_list=VAL_LIST,
)

print(f"val_len: {len(val_dataset_list)}")

cfg.ml.num_epochs = 10
cfg.ml.batch_size = 44
cfg.ml.test_batch_size = 44
cfg.ml.num_workers = 4
cfg.ml.accumulate_grad_num = 1
cfg.ml.grad_clip_val = 1.0
cfg.ml.check_val_every_n_steps = 150
cfg.ml.mix_precision = "32"

cfg.ml.optimizer.optimizer_name = "adamw"
cfg.ml.optimizer.ssl_lr = 2e-5
cfg.ml.optimizer.head_lr = 1e-4
cfg.ml.optimizer.warmup_epoch = 2  # 全エポックの1割くらい
cfg.ml.optimizer.num_cycles = 0.5

cfg.path.model_save_dir = f"{LOG_SAVE_DIR}/ckpt"
cfg.path.val_save_dir = f"{LOG_SAVE_DIR}/val"

# model
cfg.model.model_name = "wavlm_bilstmattention_ds"
cfg.model.w2v2.dropout = 0.3
cfg.model.w2v2.lstm_layers = 3
cfg.model.w2v2.lstm_dropout = 0.1  # lstmのドロップアウトは小さくする
cfg.model.w2v2.lstm_hidden_dim = 256
cfg.model.w2v2.is_freeze_ssl = False
cfg.model.w2v2.ds_hidden_dim = 32

cfg.model.w2v2.ds_num = 3
cfg.data.train_dataset_dict = {"track3": 0, "bvccmain": 1, "somos": 2}
cfg.data.test_dataset_dict = {"track3": 0, "bvccmain": 1, "somos": 2}

# dataset
cfg.data.max_duration = 5
cfg.data.pitch_shift_max = 150
cfg.data.time_wrap_max = 1.05
cfg.data.time_wrap_min = 0.95
cfg.data.is_label_normalize = True

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
        train_dataset_list=train_dataset_list,
        val_dataset_list=val_dataset_list,
        test_dataset_list=[],
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
