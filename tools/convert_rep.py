import argparse
import os
import pdb

import onnx
import onnxsim
import torch

from nanodet.model.arch import build_model
from nanodet.util import Logger, cfg, load_config, load_model_weight

from collections import OrderedDict

def main(config, model_path):
    logger = Logger(-1, config.save_dir, False)
    model = build_model(config.model)
    checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
    # pdb.set_trace()
    load_model_weight(model, checkpoint, logger)
    if config.model.arch.backbone.name == "RepVGG":
        deploy_config = config.model
        deploy_config.arch.backbone.update({"deploy": True})
        deploy_model = build_model(deploy_config)
        from nanodet.model.backbone.repvgg import repvgg_det_model_convert

        model = repvgg_det_model_convert(model, deploy_model)

    state_dict = model.state_dict()
    data = {}
    new_satet_dict = OrderedDict()
    for k, v in state_dict.items():
        k = "model." + k
        new_satet_dict[k] = v
    data["state_dict"] = new_satet_dict

    torch.save(data, model_path[:-5] + "_deploy.ckpt")

def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Convert .pth or .ckpt model to onnx.",
    )
    parser.add_argument("--cfg_path", type=str, help="Path to .yml config file.")
    parser.add_argument(
        "--model_path", type=str, default=None, help="Path to .ckpt model."
    )
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    cfg_path = args.cfg_path
    model_path = args.model_path
    load_config(cfg, cfg_path)
    main(cfg, model_path)