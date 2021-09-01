import random

from config import config
from task_tree import task_tree


class task_forest:
    def __init__(self, index, training_set):
        self.index = index
        self.training_set = training_set
        self.tree_list = []

    def gen_trees(self):
        f = open(self.training_set)
        file_list = f.readlines()
        random.shuffle(file_list)
        f.close()
        for file_index in range(len(file_list)):
            file_dir = file_list[file_index].strip("\n")
            tree = task_tree(index=file_index, data_set=[file_dir], model=None, attribute=None)
            self.tree_list.append(tree)
        print(len(self.tree_list))


if __name__ == "__main__":
    data_dir = config.training_set
    forest = task_forest(0, data_dir)
    forest.gen_trees()
