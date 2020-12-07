#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  6 15:18:55 2020

@author: edward
"""

import os, shutil, sys
import glob
import random

random.seed(0)

os.makedirs("data/train/train/", exist_ok=True)
os.makedirs("data/train/val/", exist_ok=True)

images= sorted(glob.glob("data/train/*.jpg"))
labels = sorted(glob.glob("data/train/*.npy"))
files = list(zip(images, labels))
                
random.shuffle(files)
split = 0.2

total_train = int(len(files)*(1.0-split))

dir_sep = "\\" if sys.platform == "win32" else "/"

print(f"files={files[:10]}")
print(f"total_train={total_train}")

print("Start loop 1")

for image_file, label_file in files[:total_train]:
    shutil.move(image_file, image_file.replace(f"train{dir_sep}", f"train{dir_sep}train{dir_sep}"))
    shutil.move(label_file, label_file.replace(f"train{dir_sep}", f"train{dir_sep}train{dir_sep}"))
print("End loop 1")

print("Start loop 2")
for image_file, label_file in files[total_train:]:
    shutil.move(image_file, image_file.replace(f"train{dir_sep}", f"train{dir_sep}val{dir_sep}"))
    shutil.move(label_file, label_file.replace(f"train{dir_sep}", f"train{dir_sep}val{dir_sep}"))
print("End loop 2")
