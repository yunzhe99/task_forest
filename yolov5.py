'''
this file is used for yolov5 support in calam
'''

import os
from sh import python, cd, ls
from config import config


def yolov5_train(train_data, model_name, base_weight=None):
    cd("../yolov5s/")
    python("train.py", "--weights", base_weight, "--data", train_data, "--epochs", config.epoch_nume, "--name", model_name, "--batch-size", config.batch_size, "--device", config.device)

if __name__ == "__main__":
    print(ls())
