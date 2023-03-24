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
import json
import sys
import tensorflow as tf
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import TQDMProgressBar

import torch
import torch.nn as nn
import onnx
import onnxsim
import numpy as np 
import random
sys.path.insert(0, '/media/tclwh2/facepro/lg/nanodet_9652')

from tinynn.graph.quantization.quantizer import QATQuantizer
from tinynn.converter import TFLiteConverter
from tinynn.graph.quantization.quantizer import PostQuantizer
from tinynn.graph.tracer import model_tracer, trace
from tinynn.util.cifar10 import get_dataloader, calibrate_zcc
from tinynn.util.train_util import DLContext, get_device
from tinynn.graph.quantization.cross_layer_equalization import cross_layer_equalize

from nanodet.model.arch import build_model

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
    parser.add_argument("--seed", type=int, default=999, help="random seed")
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
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(args.seed)
        random.seed(args.seed)

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
                      # +train_dataset18+train_dataset19
    all_train_dataset = train_dataset1
    # imgs_name = {}
    # imgs_name["name"] = []
    
    # quant_imgs = random.sample(range(0, 81432), 300)

    # for one in quant_imgs:
    #     img_name = all_train_dataset[one]["img_info"]["file_name"]
    #     imgs_name["name"].append(img_name)
    # with open("quant_imgs.json", "w") as f:
    #     json.dump(imgs_name, f)
    # print("finish!!")
    # pdb.set_trace()



    evaluator = build_evaluator(cfg.evaluator, val_dataset)

    train_dataloader = torch.utils.data.DataLoader(
        all_train_dataset,
        batch_size=96,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        collate_fn=naive_collate,
        drop_last=True,
    )
    val_dataloader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=96,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
        collate_fn=naive_collate,
        drop_last=False,
    )

    logger.info("Creating model...")
    work_dir="./out_repvgg"
    onnx_name = "repvgg_normal_pan_k5_nearest_1cle_G.onnx"
    with model_tracer():
        model = build_model(cfg.model)
        print(model)
        # ckpt权重路径
        checkpoint = torch.load("/model_best/model_best.ckpt", map_location=lambda storage, loc: storage)
        load_model_weight(model, checkpoint, logger)
        dummy_input = torch.randn((1, 3, 288, 512))
        if cfg.model.arch.backbone.name == "RepVGG":
            deploy_config = cfg.model
            deploy_config.arch.backbone.update({"deploy": True})
            deploy_model = build_model(deploy_config)
            from nanodet.model.backbone.repvgg import repvgg_det_model_convert
            model = repvgg_det_model_convert(model, deploy_model)

        cross_layer_equalize(model, dummy_input)
        # cross_layer_equalize(model, dummy_input)
        # cross_layer_equalize(model, dummy_input)
        # cross_layer_equalize(model, dummy_input)
        torch.onnx.export(
        model,
        dummy_input,
        onnx_name,
        verbose=False,
        keep_initializers_as_inputs=True,
        opset_version=11,
        input_names=["data"],
        output_names=["output"],
        )
        input_data = {"data": dummy_input.detach().cpu().numpy()}
        model_sim, flag = onnxsim.simplify(onnx_name, input_data=input_data)
        onnx.save(model_sim, onnx_name)


        quantizer = PostQuantizer(model, dummy_input,
                                  config={
                                      # 'rewrite_graph': False,
                                          'asymmetric': True,
                                          'per_tensor': True,
                                          "disable_requantization_for_cat" : True,
                                          'backend': 'qnnpack',
                                          'algorithm': 'l2',
                                          "quantized_input_stats": [(0., 255.)]}
                                  , work_dir=work_dir)
        qat_model = quantizer.quantize()

        # graph = trace(model, dummy_input)

        # graph.generate_code(f'nanodet.py', f'nanodet.pth', 'nanodet')
        #generate qat model file and weights

    #
    # Use DataParallel to speed up calibrating when possible
    if torch.cuda.device_count() > 1:
        qat_model = nn.DataParallel(qat_model)

    # Move model to the appropriate device
    device = get_device()
    # print(device)
    # pdb.set_trace()
    qat_model.to(device=device)
    context = DLContext()
    context.device = device
    context.train_loader, context.val_loader = train_dataloader,val_dataloader
    context.max_iteration = 4

    # Post quantization calibration
    calibrate_zcc(qat_model, context)

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
                                    tflite_path=work_dir+'/qat_model_repvgg_0cle.tflite',
                                    quantize_target_type='uint8',
                                    fuse_quant_dequant=True,
                                    # enable_mtk_ops=True
                                    # group_conv_rewrite=True,
                                    # rewrite_quantizable=True
                                    )

        converter.convert()

if __name__ == "__main__":
    args = parse_args()
    main(args)
