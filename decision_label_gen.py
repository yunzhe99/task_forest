import os
import json

import numpy as np

from tqdm import tqdm
from tree import fusion_inference, Tree, tree_fusion
from forest_train_bdd import tree_initial
from tools import get_md5
from mmdet.datasets import build_dataloader, build_dataset
from mmdet.apis import init_detector, inference_detector


human_task_list = [
            ['000', '010', '020', '100', '110', '120', '200', '210', '220', '210', '220', '300', '310', '320', '400', '410', '420', '500', '510', '520', '600', '610', '620', '630'],
            ['002', '012', '022', '112', '122', '202', '212', '222', '302', '312', '322', '402', '412', '422', '502', '512', '522', '602', '612', '622', '632'],
            ['003', '013', '023', '213', '313', '413', '503', '513', '523', '613', '633'],
            ['004', '014', '024', '034', '114', '204', '214', '224', '304', '314', '324', '404', '414', '424', '504', '514', '524', '604', '614', '624', '634'],
            ['010', '012', '013', '014', '016'],
            ['000', '002', '003', '004', '006', '300', '302', '304', '200', '202', '204', '310', '312', '313', '314', '210', '211', '212', '213', '214', '216'],
            ['020', '021', '022', '023', '024', '025', '026', '320', '322', '324', '220', '222', '224'],
            ['510', '512', '513', '514', '410', '411', '412', '413', '414', '415', '110', '112', '114', '500', '502', '503', '504', '400', '402', '404', '100'],
            ['520', '522', '523', '524', '526', '420', '422', '424', '426', '120', '122'],
            ['000', '000', '002', '002', '003', '003', '004', '004', '006', '010', '010', '012', '012', '013', '013', '014', '014', '016', '020', '020', '021', '022', '022', '023', '023', '024', '024', '025', '026', '034', '100', '100', '110', '110', '112', '112', '114', '114', '120', '120', '122', '122', '200', '200', '202', '202', '204', '204', '210', '210', '210', '211', '212', '212', '213', '213', '214', '214', '216', '220', '220', '220', '222', '222', '224', '224', '300', '300', '302', '302', '304', '304', '310', '310', '312', '312', '313', '313', '314', '314', '320', '320', '322', '322', '324', '324', '400', '400', '402', '402', '404', '404', '410', '410', '411', '412', '412', '413', '413', '414', '414', '415', '420', '420', '422', '422', '424', '424', '426', '500', '500', '502', '502', '503', '503', '504', '504', '510', '510', '512', '512', '513', '513', '514', '514', '520', '520', '522', '522', '523', '523', '524', '524', '526', '600', '602', '604', '610', '612', '613', '614', '620', '622', '624', '630', '632', '633', '634'],
            ['000', '002', '003', '004', '010', '010', '012', '012', '013', '013', '014', '014', '016', '020', '020', '021', '022', '022', '023', '023', '024', '024', '025', '026', '034', '100', '100', '110', '110', '112', '112', '114', '114', '120', '120', '122', '122', '200', '202', '204', '210', '210', '212', '213', '214', '220', '220', '220', '222', '222', '224', '224', '300', '302', '304', '310', '312', '313', '314', '320', '320', '322', '322', '324', '324', '400', '400', '402', '402', '404', '404', '410', '410', '411', '412', '412', '413', '413', '414', '414', '415', '420', '420', '422', '422', '424', '424', '426', '500', '500', '502', '502', '503', '503', '504', '504', '510', '510', '512', '512', '513', '513', '514', '514', '520', '520', '522', '522', '523', '523', '524', '524', '526', '600', '602', '604', '610', '612', '613', '614', '620', '622', '624', '630', '632', '633', '634'],
            ['004', '014', '024', '034', '114', '204', '214', '224', '304', '314', '324', '404', '414', '424', '504', '514', '524', '604', '614', '624', '634'],
            ['000', '002', '003', '004', '006', '200', '202', '204', '210', '211', '212', '213', '214', '216', '300', '302', '304', '310', '312', '313', '314'],
            ['000', '002', '003', '010', '010', '012', '012', '013', '013', '014', '016', '020', '020', '021', '022', '022', '023', '023', '024', '025', '026', '100', '100', '110', '110', '112', '112', '114', '120', '120', '122', '122', '200', '202', '210', '210', '212', '213', '220', '220', '220', '222', '222', '224', '300', '302', '310', '312', '313', '320', '320', '322', '322', '324', '400', '400', '402', '402', '404', '410', '410', '411', '412', '412', '413', '413', '414', '415', '420', '420', '422', '422', '424', '426', '500', '500', '502', '502', '503', '503', '504', '510', '510', '512', '512', '513', '513', '514', '520', '520', '522', '522', '523', '523', '524', '526', '600', '602', '610', '612', '613', '620', '622', '630', '632', '633'],
            ['000', '002', '003', '010', '010', '012', '012', '013', '013', '014', '016', '020', '020', '021', '022', '022', '023', '023', '024', '025', '026', '100', '110', '112', '120', '120', '122', '122', '200', '202', '210', '210', '212', '213', '220', '220', '220', '222', '222', '224', '300', '302', '310', '312', '313', '320', '320', '322', '322', '324', '400', '402', '410', '412', '413', '420', '420', '422', '422', '424', '426', '500', '502', '503', '510', '512', '513', '520', '520', '522', '522', '523', '523', '524', '526', '600', '602', '610', '612', '613', '620', '622', '630', '632', '633'],
            ['000', '003', '010', '010', '012', '013', '013', '014', '016', '020', '020', '021', '022', '023', '023', '024', '025', '026', '100', '110', '120', '120', '122', '200', '210', '210', '213', '220', '220', '220', '222', '224', '300', '310', '313', '320', '320', '322', '324', '400', '410', '413', '420', '420', '422', '424', '426', '500', '503', '510', '513', '520', '520', '522', '523', '523', '524', '526', '600', '610', '613', '620', '630', '633'],
            ['000', '003', '010', '013', '020', '023', '100', '110', '120', '120', '122', '200', '210', '210', '213', '220', '220', '300', '310', '313', '320', '400', '410', '413', '420', '420', '422', '424', '426', '500', '503', '510', '513', '520', '520', '522', '523', '523', '524', '526', '600', '610', '613', '620', '630', '633'],
            ['000', '003', '010', '013', '020', '023', '100', '110', '120', '200', '210', '210', '213', '220', '220', '300', '310', '313', '320', '400', '410', '413', '420', '500', '503', '510', '513', '520', '523', '600', '610', '613', '620', '630', '633']
        ]


def list_md5_weight(task_list=human_task_list):
    checkpoint_list = []
    for human_task in task_list:
        task_index = ''
        human_task.sort()
        for task_each in human_task:
            task_index = task_index+task_each+'-'
        task_index = task_index[:-1]
        print(task_index)
        checkpoint_list.append(os.path.join('/mnt/disk2/models_bdd', get_md5(task_index), 'epoch_50.pth'))
        print(os.path.exists(os.path.join('/mnt/disk2/models_bdd', get_md5(task_index), 'epoch_50.pth')))
    return checkpoint_list


def weight_list_gen():

    # each individual task
    tree_list = tree_initial(training=False)

    checkpoint_list = []

    for task_index in range(len(human_task_list)):
        for tree_each in tree_list:
            # print(tree_each.index)
            if tree_each.index[0] == human_task_list[task_index][0]:
                tree_final = tree_each
        
        for tree_each in tree_list:
            for tree_index in human_task_list[task_index][1:]:
                if tree_each.index[0] == tree_index:
                    tree_final = tree_fusion(tree_final, tree_each)
        
        checkpoint_path = os.path.join(tree_final.model_dir, 'epoch_50.pth')
        print(os.path.exists(checkpoint_path))
        checkpoint_list.append(checkpoint_path)
    
    # all task

    all_data_set = []
    for human_task in human_task_list:
        for node_each in human_task:
            all_data_set.append(node_each)
    
    # print(all_data_set)

    list_node = list(set(all_data_set))
    list_node.sort()
    # print(list_node)

    for tree_each in tree_list:
        # print(tree_each.index)
        if tree_each.index[0] == list_node[0]:
            tree_all = tree_each
    
    for tree_each in tree_list:
        for tree_index in list_node[1:]:
            if tree_each.index[0] == tree_index:
                tree_all = tree_fusion(tree_all, tree_each)
    checkpoint_list.append(os.path.join(tree_all.model_dir, 'epoch_50.pth'))
    print(os.path.exists(os.path.join(tree_all.model_dir, 'epoch_50.pth')))
    
    return checkpoint_list


def performance_evaluation_map(data_path, model_list, cfg='configs/bdd/yolox_nano_8x8_300e_base2_bdd.py'):
    print(data_path)
    tree = Tree(train_set='data/cityscapes/json/train/aachen.json', test_set=data_path, model_dir='models/aachen', base_cfg=cfg)

    tree.cfg.data.test.img_prefix = 'data/bdd_soda_traffic/train/images/'

    metric = tree.fusion_test(model_list)
    return metric


def label_gen_all_label_multi_label_best(data_dir='/mnt/disk2/detection_ped_car/json/train/bdd_train/', 
                                        weight_list=None,
                                        cfg='configs/bdd/yolox_nano_8x8_300e_base2_bdd.py'):

    model_list = []
    for checkpoint_file in weight_list:
        # 根据配置文件和 checkpoint 文件构建模型
        model = init_detector(cfg, checkpoint_file, device='cpu')
        model_list.append(model)

    # dataset = build_dataset(cfg.data.test)
    # data_loader = build_dataloader(
    #     dataset,
    #     samples_per_gpu=1,
    #     workers_per_gpu=1,
    #     dist=False,
    #     shuffle=False)

    f = open('forest/model_log/selected_models_bdd_train.txt', 'w')

    data_list = os.listdir(data_dir)
    # print(data_list[:10])
        
    for data_item in tqdm(data_list):
        data_item_path = os.path.join(data_dir, data_item)
        best_index_list = []
        performance_list = []
        for model_index in range(len(model_list)):
            try:
                performance = performance_evaluation_map(data_item_path, [model_list[model_index]], cfg=cfg)
            except:
                performance = 0
            performance_list.append(performance)
        for performance_index in range(len(performance_list)):
            if performance_list[performance_index] == max(performance_list):
                best_index_list.append(performance_index)
        
        for best_index in best_index_list:
            f.write(get_data_path(data_item_path) + ' ' + str(best_index) + '\n')

    f.close()


def get_data_path(data_json):
    with open(data_json,'r') as load_f:
        load_dict = json.load(load_f)
    return os.path.abspath(os.path.join('data/bdd_soda_traffic/train/images/', load_dict['images'][0]['file_name']))


# def adaptive_sampling(data_txt='/mnt/disk/detection_ped_car/data/bdd/video_test0_correct_7w5_shuffle_train.txt', 
#                     save_txt='algorithm/labels/0304_bdd_adaptive_correct_7w5_sampling_6w_conf_6_bdd_train.txt',
#                     cfg='configs/bdd/yolox_nano_8x8_300e_base2_bdd.py',
#                     cluster_length=1000,
#                     total_length=60000,
#                     sample_number=60000,
#                     p_value=0.01
#                     ):
#     model_list = []
#     for checkpoint_file in weight_list:
#         # 根据配置文件和 checkpoint 文件构建模型
#         model = init_detector(cfg, checkpoint_file, device='cpu')
#         model_list.append(model)

#     f = open(save_txt, 'w')

#     with open(data_txt, 'r') as data_list:
#         data_path_list = []
#         for data_path in data_list:
#             data_path_list.append(data_path.strip('\n'))

#     data_path_list = np.array(data_path_list[:total_length]).reshape(-1, cluster_length)

#     cluster_num = data_path_list.shape[0]

#     alpha_list = [1 for _ in range(cluster_num)]
#     beta_list = [1 for _ in range(cluster_num)]

#     sampling_count_model = np.array([0 for _ in range(len(model_list))])
#     sampling_count_cluster = np.array([0 for _ in range(cluster_num)])

#     for sample_index in tqdm(range(sample_number)):
#         choose_rate_max_index = 0
#         choose_rate_max = 0
#         # choose the cluster with sampling
#         for cluster_index in range(cluster_num):
#             # avoid multi sampling
#             if sampling_count_cluster[cluster_index] > math.log(1-math.pow((1-p_value), 1/cluster_length)) / math.log(1-1/cluster_length):
#                 choose_rate = 0
#             else:
#                 choose_rate = np.random.beta(alpha_list[cluster_index], beta_list[cluster_index])
#             if choose_rate_max < choose_rate:
#                 choose_rate_max = choose_rate
#                 choose_rate_max_index = cluster_index

#         # print('choose_rate_max', choose_rate_max)
#         # print('choose_rate_max_index', choose_rate_max_index)
#         sampling_count_cluster[choose_rate_max_index] += 1
#         # print('sampling_count_cluster', sampling_count_cluster)

#         cluster_sample_index = random.randint(0, cluster_length-1)
#         # print('cluster_sample_index', cluster_sample_index)
#         data_path = data_path_list[choose_rate_max_index][cluster_sample_index]
#         best_index_list = []
#         performance_list = []
#         for model_index in range(len(model_list)):
#             performance = performance_evaluation_map_teacher(data_path, teacher_model, model_list[model_index])
#             performance_list.append(performance)
#         for performance_index in range(len(performance_list)):
#             if performance_list[performance_index] == max(performance_list):
#                 best_index_list.append(performance_index) 
        
#         for best_index in best_index_list:
#             sampling_count_model[best_index] += 1
#             f.write(data_path + ' ' + str(best_index) + '\n')

#         if np.argmin(sampling_count_model) in best_index_list:
#             alpha_list[choose_rate_max_index] += 1
#         else:
#             beta_list[choose_rate_max_index] += 1

#     f.close()
    
#     print('sampling_count_model', sampling_count_model)
#     print('alpha_list', alpha_list)
#     print('beta_list', beta_list)
#     print('sampling_count_cluster', sampling_count_cluster)


if __name__ == "__main__":
    weight_list = list_md5_weight()
    # weight_list = weight_list_gen()
    # print(weight_list)
    label_gen_all_label_multi_label_best(weight_list=weight_list)
    # get_data_path('/mnt/disk2/detection_ped_car/json/train/bdd_train/1023.json')
