import os
import logging
import random

from tree import Tree, tree_fusion
from tools import get_md5

logging.basicConfig(level=logging.DEBUG)


def tree_initial(training=True):
    tree_list = []
    initial_task_name_list_train = os.listdir('data/bdd_soda_traffic/annotations/sub_bdd/train')
    initial_task_name_list_train.sort()
    initial_task_name_list = [initial_task_name[:-5] for initial_task_name in initial_task_name_list_train]
    
    initial_task_name_list = ['000', '002', '003', '004', '006', 
                            '010', '012', '013', '014', '016',
                            '020', '021', '022', '023', '024', '025', '026',
                            '034',
                            '100', 
                            '110', '112', '114', 
                            '120', '122',
                            '200', '202', '204',
                            '210', '211', '212', '213', '214', '216',
                            '220', '222', '224', 
                            '300', '302', '304', 
                            '310', '312', '313', '314',
                            '320', '322', '324',  
                            '400', '402', '404', 
                            '410', '411', '412', '413', '414', '415',
                            '420', '422', '424', '426', 
                            '436',
                            '500', '502', '503', '504', 
                            '510', '512', '513', '514', 
                            '520', '522', '523', '524', '526', 
                            '600', '602', '604', 
                            '610', '612', '613', '614', '616',
                            '620', '622', '624', '625', '626', 
                            '630', '632', '633', '634', '635', '636'
                            ]
    # initial_task_name_list.reverse()

    for initial_task_name in initial_task_name_list:
        print(initial_task_name)
        train_set = os.path.join('data/bdd_soda_traffic/annotations/sub_bdd/train', get_md5(initial_task_name)+'.json')
        test_set = os.path.join('data/bdd_soda_traffic/annotations/sub_bdd/val', get_md5(initial_task_name)+'.json')
        model_dir = os.path.join('/mnt/disk2/models_bdd', get_md5(initial_task_name))

        tree = Tree(train_set, test_set, model_dir, index=[str(initial_task_name)], base_cfg='configs/bdd/yolox_nano_8x8_300e_base2_bdd.py')
        if training:
            tree.train([0])
            tree.get_scores()
            print('Scores:', tree.scores)
        tree_list.append(tree)

    return tree_list


def single_task_train():
    tree_list = tree_initial(training=False)
    # print('tree list:', len(tree_list))

    # for tree_each in tree_list:
    #     print(tree_each.index)

    tree_single = tree_list[0]
    for tree_index in range(1, len(tree_list)):
        tree_single = tree_fusion(tree_single, tree_list[tree_index])
    tree_single.train([1])
    tree_single.get_scores()
    print(tree_single.scores)
    print(tree_single.print_counts())


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


def human_defined_task_train():
    human_task_list = [
        ['000', '010', '020', '100', '110', '120', '200', '210', '220', '210', '220', '300', '310', '320', '400', '410', '420', '500', '510', '520', '600', '610', '620', '630'],
        ['002', '012', '022', '112', '122', '202', '212', '222', '302', '312', '322', '402', '412', '422', '502', '512', '522', '602', '612', '622', '632'],
        ['003', '013', '023', '213', '313', '413', '503', '513', '523', '613', '633'],
        ['004', '014', '024', '034', '114', '204', '214', '224', '304', '314', '324', '404', '414', '424', '504', '514', '524', '604', '614', '624', '634'],
        ['010', '012', '013', '014', '016'],
        ['000', '002', '003', '004', '006', '300', '302', '304', '200', '202', '204', '310', '312', '313', '314', '210', '211', '212', '213', '214', '216'],
        ['020', '021', '022', '023', '024', '025', '026', '320', '322', '324', '220', '222', '224'],
        ['510', '512', '513', '514', '410', '411', '412', '413', '414', '415', '110', '112', '114', '500', '502', '503', '504', '400', '402', '404', '100'],
        ['520', '522', '523', '524', '526', '420', '422', '424', '426', '120', '122']
    ]

    # human_task_list.reverse()

    tree_list = tree_initial(training=False)
    # print('tree list:', len(tree_list))

    # for tree_each in tree_list:
    #     print(tree_each.index)

    final_tree_list = []

    for task_index in range(len(human_task_list)):
        for tree_each in tree_list:
            # print(tree_each.index)
            if tree_each.index[0] == human_task_list[task_index][0]:
                tree_final = tree_each
        
        for tree_each in tree_list:
            for tree_index in human_task_list[task_index][1:]:
                if tree_each.index[0] == tree_index:
                    tree_final = tree_fusion(tree_final, tree_each)
        # print(tree_final.index)
        tree_final.train([0])
        tree_final.get_scores()
        # print(tree_final.scores)
        # print(tree_final.print_counts())
        final_tree_list.append(tree_final)

    for tree_each in final_tree_list:
        print(tree_each.index, tree_each.scores, tree_each.print_counts())

    # tree_update(final_tree_list)


def human_defined_all_data_train():
    human_task_list = [
        ['000', '010', '020', '100', '110', '120', '200', '210', '220', '210', '220', '300', '310', '320', '400', '410', '420', '500', '510', '520', '600', '610', '620', '630'],
        ['002', '012', '022', '112', '122', '202', '212', '222', '302', '312', '322', '402', '412', '422', '502', '512', '522', '602', '612', '622', '632'],
        ['003', '013', '023', '213', '313', '413', '503', '513', '523', '613', '633'],
        ['004', '014', '024', '034', '114', '204', '214', '224', '304', '314', '324', '404', '414', '424', '504', '514', '524', '604', '614', '624', '634'],
        ['010', '012', '013', '014', '016'],
        ['000', '002', '003', '004', '006', '300', '302', '304', '200', '202', '204', '310', '312', '313', '314', '210', '211', '212', '213', '214', '216'],
        ['020', '021', '022', '023', '024', '025', '026', '320', '322', '324', '220', '222', '224'],
        ['510', '512', '513', '514', '410', '411', '412', '413', '414', '415', '110', '112', '114', '500', '502', '503', '504', '400', '402', '404', '100'],
        ['520', '522', '523', '524', '526', '420', '422', '424', '426', '120', '122']
    ]

    all_data_set = []
    for human_task in human_task_list:
        for node_each in human_task:
            all_data_set.append(node_each)
    
    print(all_data_set)

    list_node = list(set(all_data_set))
    list_node.sort()
    print(list_node)

    tree_list = tree_initial(training=False)
    # print('tree list:', len(tree_list))

    # for tree_each in tree_list:
    #     print(tree_each.index)

    for tree_each in tree_list:
        # print(tree_each.index)
        if tree_each.index[0] == list_node[0]:
            tree_final = tree_each
    
    for tree_each in tree_list:
        for tree_index in list_node[1:]:
            if tree_each.index[0] == tree_index:
                tree_final = tree_fusion(tree_final, tree_each)

    print(tree_final.index)
    tree_final.train([1])
    tree_final.get_scores()

    print(tree_final.scores)
    print(tree_final.print_counts())


def human_defined_all_data_train_x_model():
    tree_all = Tree(train_set='data/bdd_soda_traffic/annotations/sub_bdd/train/8b5306e3d771e16ad901c1fb4ebe4123.json', test_set='data/bdd_soda_traffic/annotations/sub_bdd/val/8b5306e3d771e16ad901c1fb4ebe4123.json', model_dir='all_model', base_cfg='configs/bdd/yolox_x_8x2_300e_all3.py')
    tree_all.train([1])
    tree_all.get_scores()

    print(tree_all.scores)
    print(tree_all.print_counts())
    

def grouped_task_train():
    grouped_task_list = [
        ['000', '000', '002', '002', '003', '003', '004', '004', '006', '010', '010', '012', '012', '013', '013', '014', '014', '016', '020', '020', '021', '022', '022', '023', '023', '024', '024', '025', '026', '034', '100', '100', '110', '110', '112', '112', '114', '114', '120', '120', '122', '122', '200', '200', '202', '202', '204', '204', '210', '210', '210', '211', '212', '212', '213', '213', '214', '214', '216', '220', '220', '220', '222', '222', '224', '224', '300', '300', '302', '302', '304', '304', '310', '310', '312', '312', '313', '313', '314', '314', '320', '320', '322', '322', '324', '324', '400', '400', '402', '402', '404', '404', '410', '410', '411', '412', '412', '413', '413', '414', '414', '415', '420', '420', '422', '422', '424', '424', '426', '500', '500', '502', '502', '503', '503', '504', '504', '510', '510', '512', '512', '513', '513', '514', '514', '520', '520', '522', '522', '523', '523', '524', '524', '526', '600', '602', '604', '610', '612', '613', '614', '620', '622', '624', '630', '632', '633', '634'],
        ['000', '002', '003', '004', '010', '010', '012', '012', '013', '013', '014', '014', '016', '020', '020', '021', '022', '022', '023', '023', '024', '024', '025', '026', '034', '100', '100', '110', '110', '112', '112', '114', '114', '120', '120', '122', '122', '200', '202', '204', '210', '210', '212', '213', '214', '220', '220', '220', '222', '222', '224', '224', '300', '302', '304', '310', '312', '313', '314', '320', '320', '322', '322', '324', '324', '400', '400', '402', '402', '404', '404', '410', '410', '411', '412', '412', '413', '413', '414', '414', '415', '420', '420', '422', '422', '424', '424', '426', '500', '500', '502', '502', '503', '503', '504', '504', '510', '510', '512', '512', '513', '513', '514', '514', '520', '520', '522', '522', '523', '523', '524', '524', '526', '600', '602', '604', '610', '612', '613', '614', '620', '622', '624', '630', '632', '633', '634'],
        ['004', '014', '024', '034', '114', '204', '214', '224', '304', '314', '324', '404', '414', '424', '504', '514', '524', '604', '614', '624', '634'],
        ['000', '002', '003', '004', '006', '200', '202', '204', '210', '211', '212', '213', '214', '216', '300', '302', '304', '310', '312', '313', '314'],
        ['000', '002', '003', '010', '010', '012', '012', '013', '013', '014', '016', '020', '020', '021', '022', '022', '023', '023', '024', '025', '026', '100', '100', '110', '110', '112', '112', '114', '120', '120', '122', '122', '200', '202', '210', '210', '212', '213', '220', '220', '220', '222', '222', '224', '300', '302', '310', '312', '313', '320', '320', '322', '322', '324', '400', '400', '402', '402', '404', '410', '410', '411', '412', '412', '413', '413', '414', '415', '420', '420', '422', '422', '424', '426', '500', '500', '502', '502', '503', '503', '504', '510', '510', '512', '512', '513', '513', '514', '520', '520', '522', '522', '523', '523', '524', '526', '600', '602', '610', '612', '613', '620', '622', '630', '632', '633'],
        ['000', '002', '003', '010', '010', '012', '012', '013', '013', '014', '016', '020', '020', '021', '022', '022', '023', '023', '024', '025', '026', '100', '110', '112', '120', '120', '122', '122', '200', '202', '210', '210', '212', '213', '220', '220', '220', '222', '222', '224', '300', '302', '310', '312', '313', '320', '320', '322', '322', '324', '400', '402', '410', '412', '413', '420', '420', '422', '422', '424', '426', '500', '502', '503', '510', '512', '513', '520', '520', '522', '522', '523', '523', '524', '526', '600', '602', '610', '612', '613', '620', '622', '630', '632', '633'],
        ['000', '003', '010', '013', '020', '020', '021', '022', '023', '023', '024', '025', '026', '100', '110', '120', '120', '122', '200', '210', '210', '213', '220', '220', '220', '222', '224', '300', '310', '313', '320', '320', '322', '324', '400', '410', '413', '420', '420', '422', '424', '426', '500', '503', '510', '513', '520', '520', '522', '523', '523', '524', '526', '600', '610', '613', '620', '630', '633']
    ]

    # human_task_list.reverse()

    tree_list = tree_initial(training=False)
    # print('tree list:', len(tree_list))

    # for tree_each in tree_list:
    #     print(tree_each.index)

    final_tree_list = []

    for task_index in range(len(grouped_task_list)):
        for tree_each in tree_list:
            # print(tree_each.index)
            if tree_each.index[0] == grouped_task_list[task_index][0]:
                tree_final = tree_each
        
        for tree_each in tree_list:
            for tree_index in grouped_task_list[task_index][1:]:
                if tree_each.index[0] == tree_index:
                    tree_final = tree_fusion(tree_final, tree_each)
        # print(tree_final.index)
        tree_final.train([0])
        tree_final.get_scores()
        # print(tree_final.scores)
        # print(tree_final.print_counts())
        final_tree_list.append(tree_final)

    for tree_each in final_tree_list:
        print(tree_each.index, tree_each.scores, tree_each.print_counts())

    # tree_update(final_tree_list)


if __name__ == '__main__':
    # single_task_train()

    tree_main()

    # human_defined_task_train()

    # grouped_task_train()

    # human_defined_all_data_train_x_model()
