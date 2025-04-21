"""Wav2vec2 linear training script."""

import shutil

import torch
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.loggers import WandbLogger

from track3.core.config import Config
from track3.core.dataset.mospl import MOSDataModule
from track3.core.models.sslpl import MOSPredictorModule
from track3.exp.utils import CheckpointEveryEpoch, get_dataset_fp_label_userid, get_user_map

cfg = Config()
seed_everything(cfg.ml.seed)
##########
# Params #
##########

VERSION = "00001"
EXP_ID = "sslw2v2_linear"
WANDB_PROJECT_NAME = "moschallenge2025track3"
IS_LOGGING = True
FAST_DEV_RUN = False

LOG_SAVE_DIR = f"logs/{EXP_ID}/v{VERSION}"
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

user_map = get_user_map(user_csv_fp)
print(f"user_len: {len(user_map)}")
train_audio_list, train_label_list, train_user_id_list = get_dataset_fp_label_userid(
    dataset_list=TRAIN_LIST,
    user_map=user_map,
)
print(f"train_len: {len(train_audio_list)}")
val_audio_list, val_label_list, val_user_id_list = get_dataset_fp_label_userid(
    dataset_list=VAL_LIST,
    user_map=user_map,
)
print(f"val_len: {len(val_audio_list)}")
test_audio_list, test_label_list, test_user_id_list = get_dataset_fp_label_userid(
    dataset_list=TEST_LIST,
    user_map=user_map,
)
print(f"test_len: {len(test_audio_list)}")
##########
cfg.ml.num_epochs = 10
cfg.ml.batch_size = 16
cfg.ml.test_batch_size = 16
cfg.ml.num_workers = 2
cfg.ml.accumulate_grad_num = 1
cfg.ml.grad_clip_val = 500
cfg.ml.check_val_every_n_epoch = 1
cfg.ml.mix_precision = "bf16"

cfg.ml.optimizer.optimizer_name = "adamw"
cfg.ml.optimizer.lr = 1e-4
cfg.ml.optimizer.weight_decay = 1e-4
cfg.ml.optimizer.adam_epsilon = 1e-4
cfg.ml.optimizer.warmup_epoch = 1
cfg.ml.optimizer.num_cycles = 0.5

cfg.path.model_save_dir = f"{LOG_SAVE_DIR}/ckpt"
cfg.path.val_save_dir = f"{LOG_SAVE_DIR}/val"

# model
cfg.model.model_name = "wav2vec2_linear"
cfg.model.w2v2.ssl_out_dim = 768
cfg.model.w2v2.pretrained_model_name = "facebook/wav2vec2-base-960h"
cfg.model.w2v2.dropout = 0.1
cfg.model.w2v2.is_freeze_ssl = False


def train() -> None:
    """Train the model."""
    dataset = MOSDataModule(
        config=cfg,
        train_audio_list=train_audio_list,
        train_label_list=train_label_list,
        train_user_id_list=train_user_id_list,
        val_audio_list=val_audio_list,
        val_label_list=val_label_list,
        val_user_id_list=val_user_id_list,
        test_audio_list=test_audio_list,
        test_label_list=test_label_list,
        test_user_id_list=test_user_id_list,
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
        LearningRateMonitor(logging_interval="epoch"),
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
