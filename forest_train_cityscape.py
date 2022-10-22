import os
import logging

from tree import Tree, tree_fusion
from tools import task_name_get

logging.basicConfig(level=logging.DEBUG)


def tree_initial():
    tree_list = []
    initial_task_name_list = ['aachen', 'bochum', 'bremen', 'cologne', 'darmstadt', 'dusseldorf',
                            'erfurt', 'hamburg', 'hanover', 'jena', 'krefeld', 'monchengladbach',
                            'strasbourg', 'stuttgart', 'tubingen', 'ulm', 'weimar', 'zurich']
    # initial_task_name_list.reverse()

    for initial_task_name in initial_task_name_list:
        print(initial_task_name)
        train_set = os.path.join('data/cityscapes/json/train', initial_task_name+'.json')
        test_set = os.path.join('data/cityscapes/json/val', initial_task_name+'.json')
        model_dir = os.path.join('models', initial_task_name)

        tree = Tree(train_set, test_set, model_dir, index=[str(initial_task_name)])
        tree.train([0])
        tree.get_scores()
        tree_list.append(tree)

    return tree_list


def single_task_train():
    tree_list = tree_initial()
    print('tree list:', len(tree_list))

    for tree_each in tree_list:
        print(tree_each.index)

    tree_single = tree_list[0]
    for tree_index in range(1, len(tree_list)):
        tree_single = tree_fusion(tree_single, tree_list[tree_index])
    tree_single.train()
    tree_single.get_scores()
    print(tree_single.scores)


def check_fusion_type(tree_fused, tree1, tree2):
    if tree_fused.scores > max(tree1.scores, tree2.scores):
        return 0  # entire benefical
    elif tree_fused.scores > tree1.scores:
        return 1  # self benefical on tree1
    elif tree_fused.scores > tree2.scores:
        return 2  # self benefical on tree2
    else:
        return 3


def try_fusion(tree_list):
    tree1 = tree_list[0]
    best_tree = None
    best_index = 1
    best_type = 3
    for tree_index in range(1, len(tree_list)):
        print(tree_index)
        tree2 = tree_list[tree_index]
        tree_fused = tree_fusion(tree1, tree2)
        tree_fused.train()
        tree_fused.get_scores()
        fusion_type = check_fusion_type(tree_fused, tree1, tree2)
        print(tree_fused.scores, fusion_type)
        if best_tree is None:
            best_tree = tree_fused
            best_type = fusion_type
        else:
            if fusion_type < best_type:
                best_tree = tree_fused
                best_index = tree_index
                best_type = fusion_type
            elif fusion_type == best_type and tree_fused.scores > best_tree.scores:
                best_tree = tree_fused
                best_index = tree_index
    logging.debug('Best Fusion: {} and {}'.format(str(0), str(best_index)))

    return best_tree, best_type, best_index


def tree_update(tree_list):
    finish_count = 0
    def take_scores(elem):
        return elem.scores
    
    while finish_count < 10:
        
        tree_list.sort(key=take_scores)
        logging.info("Before Updating!")
        for tree_each in tree_list:
            print(tree_each.index, tree_each.scores)
        tree_fused, fusion_flag, best_index = try_fusion(tree_list)
        if fusion_flag == 0:
            del tree_list[0]  # delete all two trees
            del tree_list[best_index-1]  # the index changed when the first on deleted.
            tree_list.append(tree_fused)
        elif fusion_flag == 1:
            del tree_list[0]  # delete the 2nd tree
            tree_list.append(tree_fused)
        elif fusion_flag == 2:
            del tree_list[best_index]  # delete the 1st tree
            tree_list.append(tree_fused)
        else:
            finish_count += 1

        logging.info("Tree Updating, Fusion Flag: " + str(fusion_flag))
        for tree_each in tree_list:
            print(tree_each.index, tree_each.scores)
        print()
        print()
    return tree_list


def tree_main():
    tree_list = tree_initial()

    for tree in tree_list:
        print(tree.index, tree.scores)

    tree_list = tree_update(tree_list)

    logging.info('Final Tree:')
    for tree in tree_list:
        print(tree.index, tree.scores)


if __name__ == '__main__':
    single_task_train()

    # tree_main()
