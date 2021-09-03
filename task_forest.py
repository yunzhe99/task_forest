import random

from config import config
from task_tree import task_tree


class task_forest:
    def __init__(self, forest_index, training_set):
        self.forest_index = forest_index  # the index of the forest
        self.training_set = training_set  # the whole training set of the forest
        self.tree_list = []  # the list of the trees in the forest
        self.tree_index = len(self.tree_list)  # the next index can be used

    def gen_trees(self):
        '''
        gen the trees from bottle to top: define each one as a single tree
        '''
        f = open(self.training_set)
        file_list = f.readlines()
        random.shuffle(file_list)
        f.close()
        for file_index in range(len(file_list)):
            file_dir = file_list[file_index].strip("\n")
            tree = task_tree(index=file_index + self.tree_index, data_set=[file_dir], model=None, attribute=None)
            tree.model_training()
            self.tree_list.append(tree)
        print(len(self.tree_list))

    def tree_merge(self, merge_index1, merge_index2):
        '''
        the merge operation of two trees
        '''
        tree1 = self.tree_list[merge_index1]
        tree2 = self.tree_list[merge_index2]
        tree_new = task_tree(index=self.tree_index, data_set=(tree1.data_set + tree2.data_set), model=None, attribute=None)
        tree_new.model_training()
        tree_new.attribution()
        return tree_new
    
    def single_tree(self):
        '''
        define one tree for the whole forest
        '''
        tree = task_tree(index=0 + self.tree_index, data_set=self.training_set, model=None, attribute=None)
        print("Start Model Training of single tree")
        tree.model_training()
        print("Training Finished!")


if __name__ == "__main__":
    data_dir = config.training_set
    forest = task_forest(0, data_dir)
    forest.single_tree()
