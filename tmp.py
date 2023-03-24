import shutil
import os
import json

root_dir = "/media/tclwh2/facepro/zcc/datasets/detection_datasets"
out_dir = "./quant_imgs"

with open("quant_imgs.json", "r") as f:
    imgs_name = json.load(f)
    imgs_name = imgs_name["name"]
print(imgs_name)


for root, _, files in os.walk(root_dir):
    if len(files) != 0:
        for file in files:
            if file in imgs_name:
                src = os.path.join(root, file)
                dst = os.path.join(out_dir, file)
                shutil.copy(src, dst)
                print(src)



