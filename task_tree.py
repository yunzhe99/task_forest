class task_tree:
    '''
    The tree class designed for the task definition and usage
    '''
    def __init__(self, index, data_set, model, attribute):
        '''
        index: the index of a tree
        data_set: the data_set for the tree
        model: the model of the tree
        attribute: the attribute that can represent the tree
        '''
        self.index = index
        self.data_set = data_set
        self.model = model
        self.attribute = attribute
