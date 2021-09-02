import numpy as np
import os
import yaml

from config import config
from yolov5 import yolov5_train


class task_tree:
    '''
    The tree class designed for the task definition and usage
    '''
    def __init__(self, index, data_set, model, attribute):
        '''
        index: the index of a tree
        data_set: the data_set for the tree, list
        model: the model of the tree
        attribute: the attribute that can represent the tree
        '''
        self.index = str(index)
        data_dir = os.path.join(os.getcwd(), "data", self.index + ".txt")
        with open(data_dir,"w") as f:
            f.writelines(data_set)
        self.data_set = data_dir
        self.model = model
        self.attribute = attribute

    def model_training(self):
        f = open(self.data_set)
        file_list = f.readlines()
        sample_num = len(file_list)
        if sample_num > config.min_train:
            dic = {'train':self.data_set, 'nc':config.nc, 'names':config.names, 'val':config.val}
            yamlpath = os.path.join(os.getcwd(), 'yaml_fir', self.index + ".yaml")
            with open(yamlpath, "w") as f:
                yaml.dump(dic, f)
                yolov5_train()
                

    def attribution(self):
        pass


if __name__ == "__main__":
    dic = {'train':"./data/1.txt", 'nc':config.nc, 'names':config.names}
    yamlpath = os.path.join(os.getcwd(), "test.yaml")
    print(yamlpath)
    with open(yamlpath, "w") as f:
        yaml.dump(dic, f)
