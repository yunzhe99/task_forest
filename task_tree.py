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
        data_set: the data_set for the tree, list or txt
        model: the model of the tree
        attribute: the attribute that can represent the tree
        '''
        self.index = str(index)
        if type(data_set) == list:
            data_dir = os.path.join(os.getcwd(), "txt", self.index + ".txt")
            with open(data_dir,"w") as f:
                f.writelines(data_set)
            self.data_set = data_dir
        else:
            self.data_set = data_set  # the case of txt input
        self.model = model
        self.attribute = attribute

    def model_training(self):
        '''
        use the dataset of the tree to train a model for inference
        '''
        f = open(self.data_set)
        file_list = f.readlines()
        sample_num = len(file_list)
        if sample_num > config.min_train:
            dic = {'train':self.data_set, 'nc':config.nc, 'names':config.names, 'val':config.val}
            yamlpath = os.path.join(os.getcwd(), 'yaml_dir', self.index + ".yaml")
            with open(yamlpath, "w") as f:
                yaml.dump(dic, f)  # generate the yaml file for yolov5 training
                self.model = yolov5_train(train_data=yamlpath, model_name=self.index, base_weight=self.model)

    def attribution(self):
        '''
        tree attribute extraction
        '''
        pass


if __name__ == "__main__":
    l = [1, 2, 3, 4]
    print(type(l) == list)
