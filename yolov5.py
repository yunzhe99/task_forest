'''
this file is used for yolov5 support in calam
'''

import os
from sh import python, cd, ls, conda
from config import config


def yolov5_train(train_data, model_name, base_weight=None):
    '''
    using the yolov5 file for training
    '''
    if base_weight is None:
        base_weight = 'yolov5s.pt'
    project_dir = os.path.join(os.getcwd(), "model_train", str(model_name))
    python("train.py", "--weights", base_weight, "--data", train_data, "--epochs", config.epoch_num, "--name", model_name, "--batch-size", config.batch_size, "--device", config.device, "--exist-ok", "--project", project_dir)
    model_dir = os.path.join(project_dir, "0", "weights", "best.pt")
    return model_dir


if __name__ == "__main__":
    project_dir = os.path.join(os.getcwd(), "models", str(1))
    print(project_dir)
