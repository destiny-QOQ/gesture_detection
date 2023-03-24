import pdb

from nanodet.data.dataset import build_dataset
from nanodet.evaluator import build_evaluator
from nanodet.data.dataset import build_dataset
from nanodet.util import (
    NanoDetLightningLogger,
    cfg,
    convert_old_model,
    load_config,
    mkdir,
)

import os
import json

with open("1cle_q21.json", "r") as f:
    res_json = json.load(f)


load_config(cfg, "gesture_config/repvgg_ghost_pan_512_288.yml")

val_dataset = build_dataset(cfg.data.val, "val")

evaluator = build_evaluator(cfg.evaluator, val_dataset)

# name_to_id = {}

# with open(cfg.data.val.ann_path, "r") as f:
#     data = json.load(f)
#     for one_item in data["images"]:
#         _file_name  = one_item["file_name"]
#         _id = one_item["id"]
#         assert _file_name not in name_to_id.keys()
#         name_to_id[_file_name] = _id


# with open("testdata_name_to_id.json", "w") as f1:
#     json.dump(name_to_id,f1)
# print("finish")
# pdb.set_trace()
eval_results = evaluator.evaluate(
    results=res_json, save_dir="test_data_map", rank=-1
)

print(eval_results)

