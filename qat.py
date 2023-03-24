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
import warnings
import pdb
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ProgressBar

from nanodet.data.collate import naive_collate
from nanodet.data.dataset import build_dataset
from nanodet.evaluator import build_evaluator
from nanodet.trainer.task import TrainingTask
from nanodet.util import (
    NanoDetLightningLogger,
    cfg,
    convert_old_model,
    load_config,
    load_model_weight,
    mkdir,
)
from tinynn.converter import TFLiteConverter
from tinynn.graph.quantization.quantizer import QATQuantizer
from tinynn.graph.tracer import model_tracer



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
    # pdb.set_trace()

    if args.seed is not None:
        logger.info("Set random seed to {}".format(args.seed))
        pl.seed_everything(args.seed)

    logger.info("Setting up data...")
    # train_dataset1 = build_dataset(cfg.data.train1, "train")
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
    train_dataset17 = build_dataset(cfg.data.train17, "train")
    val_dataset = build_dataset(cfg.data.val, "test")

    all_train_dataset=train_dataset17
    # +train_dataset2+train_dataset3+train_dataset4+train_dataset5+train_dataset6+ train_dataset7 + \
                      # train_dataset8 + train_dataset9 + train_dataset10+train_dataset11+train_dataset12+train_dataset13+train_dataset14+train_dataset15+train_dataset16+train_dataset17

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

    accelerator = None if len(cfg.device.gpu_ids) <= 1 else "ddp"

    with model_tracer():
        from out_qat.nanodetplus_qat import NanoDetPlus_qat
        model = NanoDetPlus_qat(num_classes=cfg.model.arch.head.num_classes,
                                loss=cfg.model.arch.head.loss,
                                input_channel=cfg.model.arch.head.input_channel,
                                feat_channels=cfg.model.arch.head.feat_channels,
                                stacked_convs=cfg.model.arch.head.stacked_convs,
                                kernel_size=cfg.model.arch.head.kernel_size,
                                strides=cfg.model.arch.head.strides,
                                conv_type="DWConv",
                                norm_cfg=dict(type="BN"),
                                reg_max=7,
                                activation="ReLU6",
                                assigner_cfg=dict(topk=13),
                                )
        model.load_state_dict(torch.load("/media/tclwh2/facepro/zhangdi/nanodet/out_deconv/nanodetplus_qat.pth"), strict=False)
        # pdb.set_trace()

        # model = task.model

        dummy_input = torch.rand((1, 3, 288, 512))

        quantizer = QATQuantizer(model, dummy_input, work_dir="out_qat",
                                 config={'rewrite_graph': False, 'per_tensor': True, 'asymmetric': True, 'backend': 'qnnpack', "quantized_input_stats" : [(0., 255.)], "disable_requantization_for_cat" : True})

        qat_model = quantizer.quantize()

        task.model = qat_model
    pdb.set_trace()
    trainer = pl.Trainer(
        default_root_dir=cfg.save_dir,
        max_epochs=cfg.schedule.total_epochs,
        gpus=cfg.device.gpu_ids,
        check_val_every_n_epoch=cfg.schedule.val_intervals,
        accelerator=accelerator,
        log_every_n_steps=cfg.log.interval,
        num_sanity_val_steps=0,
        resume_from_checkpoint=model_resume_path,
        callbacks=[ProgressBar(refresh_rate=0)],  # disable tqdm bar
        logger=logger,
        benchmark=True,
        gradient_clip_val=cfg.get("grad_clip", 0.0),
    )

    trainer.fit(task, train_dataloader, val_dataloader)

    torch.save(task.model.state_dict(), "./out_qat/last_model.pth")
    pdb.set_trace()
    qat_model.load_state_dict(torch.load("./out_qat/last_model.pth"), strict=True)

    with torch.no_grad():
        qat_model.eval()
        qat_model.cpu()

        # The step below converts the model to an actual quantized model, which uses the quantized kernels.
        qat_model = torch.quantization.convert(qat_model)

        # When converting quantized models, please ensure the quantization backend is set.
        torch.backends.quantized.engine = 'qnnpack'

        # The code section below is used to convert the model to the TFLite format
        # If you need a quantized model with a specific data type (e.g. int8)
        # you may specify `quantize_target_type='int8'` in the following line.
        # If you need a quantized model with strict symmetric quantization check (with pre-defined zero points),
        # you may specify `strict_symmetric_check=True` in the following line.
        converter = TFLiteConverter(qat_model, dummy_input,
                                    tflite_path='out_qat/qat_model.tflite',
                                    quantize_target_type='uint8',
                                    fuse_quant_dequant=True,
                                    group_conv_rewrite=True,
                                    rewrite_quantizable=True
                                    )

        converter.convert()


if __name__ == "__main__":
    args = parse_args()
    main(args)
