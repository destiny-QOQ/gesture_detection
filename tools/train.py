# Copyright 2021 RangiLyu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import sys
import warnings

import tensorflow as tf
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import TQDMProgressBar

sys.path.insert(0, '/media/tclwh2/facepro/lg/nanodet_9652')

from nanodet.data.collate import naive_collate
from nanodet.data.dataset import build_dataset
from nanodet.evaluator import build_evaluator
from nanodet.trainer.task import TrainingTask
from nanodet.util import (
    NanoDetLightningLogger,
    cfg,
    convert_old_model,
    env_utils,
    load_config,
    load_model_weight,
    mkdir,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="train config file path")
    parser.add_argument(
        "--local_rank", default=-1, type=int, help="node rank for distributed training"
    )
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    args = parser.parse_args()
    return args


def main(args):
    load_config(cfg, args.config)
    if cfg.model.arch.head.num_classes != len(cfg.class_names):
        raise ValueError(
            "cfg.model.arch.head.num_classes must equal len(cfg.class_names), "
            "but got {} and {}".format(
                cfg.model.arch.head.num_classes, len(cfg.class_names)
            )
        )
    local_rank = int(args.local_rank)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    mkdir(local_rank, cfg.save_dir)

    logger = NanoDetLightningLogger(cfg.save_dir)
    logger.dump_cfg(cfg)

    if args.seed is not None:
        logger.info("Set random seed to {}".format(args.seed))
        pl.seed_everything(args.seed)

    logger.info("Setting up data...")
    train_dataset1 = build_dataset(cfg.data.train1, "train")
    # train_dataset2 = build_dataset(cfg.data.train2, "train")
    # train_dataset3 = build_dataset(cfg.data.train3, "train")
    # train_dataset4 = build_dataset(cfg.data.train4, "train")
    # train_dataset5 = build_dataset(cfg.data.train5, "train")
    # train_dataset6 = build_dataset(cfg.data.train6, "train")
    # train_dataset7 = build_dataset(cfg.data.train7, "train")
    # train_dataset8 = build_dataset(cfg.data.train8, "train")
    # train_dataset9 = build_dataset(cfg.data.train9, "train")
    # train_dataset10 = build_dataset(cfg.data.train10, "train")
    # train_dataset11 = build_dataset(cfg.data.train11, "train")
    # train_dataset12 = build_dataset(cfg.data.train12, "train")
    # train_dataset13 = build_dataset(cfg.data.train13, "train")
    # train_dataset14 = build_dataset(cfg.data.train14, "train")
    # train_dataset15 = build_dataset(cfg.data.train15, "train")
    # train_dataset16 = build_dataset(cfg.data.train16, "train")
    # train_dataset17 = build_dataset(cfg.data.train17, "train")
    # train_dataset18 = build_dataset(cfg.data.train18, "train")
    # train_dataset19 = build_dataset(cfg.data.train19, "train")
    val_dataset = build_dataset(cfg.data.val, "test")

    # all_train_dataset=train_dataset1+train_dataset2+train_dataset3+train_dataset4+train_dataset5+train_dataset6+ train_dataset7 + \
    #                   train_dataset8 + train_dataset9 + train_dataset10+train_dataset11+train_dataset12+train_dataset13+train_dataset14+train_dataset15+train_dataset16+train_dataset17+train_dataset18
    #                   # +train_dataset18+train_dataset19
    all_train_dataset = train_dataset1
    evaluator = build_evaluator(cfg.evaluator, val_dataset)

    train_dataloader = torch.utils.data.DataLoader(
        all_train_dataset,
        batch_size=cfg.device.batchsize_per_gpu,
        shuffle=True,
        num_workers=cfg.device.workers_per_gpu,
        pin_memory=True,
        collate_fn=naive_collate,
        drop_last=True,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=cfg.device.batchsize_per_gpu,
        shuffle=False,
        num_workers=cfg.device.workers_per_gpu,
        pin_memory=True,
        collate_fn=naive_collate,
        drop_last=False,
    )

    logger.info("Creating model...")
    task = TrainingTask(cfg, evaluator)

    if "load_model" in cfg.schedule:
        ckpt = torch.load(cfg.schedule.load_model)
        if "pytorch-lightning_version" not in ckpt:
            warnings.warn(
                "Warning! Old .pth checkpoint is deprecated. "
                "Convert the checkpoint with tools/convert_old_checkpoint.py "
            )
            ckpt = convert_old_model(ckpt)
        load_model_weight(task.model, ckpt, logger)
        logger.info("Loaded model weight from {}".format(cfg.schedule.load_model))

    model_resume_path = (
        os.path.join(cfg.save_dir, "model_last.ckpt")
        if "resume" in cfg.schedule
        else None
    )
    if cfg.device.gpu_ids == -1:
        logger.info("Using CPU training")
        accelerator, devices, strategy = "cpu", None, None
    else:
        accelerator, devices, strategy = "gpu", cfg.device.gpu_ids, None

    if devices and len(devices) > 1:
        strategy = "ddp"
        env_utils.set_multi_processing(distributed=True)

    trainer = pl.Trainer(
        default_root_dir=cfg.save_dir,
        max_epochs=cfg.schedule.total_epochs,
        check_val_every_n_epoch=cfg.schedule.val_intervals,
        accelerator=accelerator,
        devices=devices,
        log_every_n_steps=cfg.log.interval,
        num_sanity_val_steps=0,
        resume_from_checkpoint=model_resume_path,
        callbacks=[TQDMProgressBar(refresh_rate=0)],  # disable tqdm bar
        logger=logger,
        benchmark=cfg.get("cudnn_benchmark", True),
        gradient_clip_val=cfg.get("grad_clip", 0.0),
        strategy=strategy,
    )

    trainer.fit(task, train_dataloader, val_dataloader)


if __name__ == "__main__":
    args = parse_args()
    main(args)
