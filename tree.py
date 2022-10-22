import os
import json
import heapq
import torch
import numpy as np

from mmcv import Config
from mmcv.ops.nms import nms
from mmcls.apis import init_model
from mmdet.apis import init_detector, inference_detector
from mmdet.datasets import build_dataloader, build_dataset
from tools import json_group, check_available_training, get_md5
from train_tree import train_one_model
from test_tree import test_one_dataset
from mmcv.parallel import collate, scatter
from mmcls.datasets.pipelines import Compose


class Tree:
    def __init__(self, train_set=None, test_set=None, model_dir=None, seed=42, base_cfg='configs/bdd/yolox_nano_8x8_300e_base2_bdd.py', index=-1):
        '''
        train_set: the json file for training
        test_set: the json file for testing
        model_dir: the dir for training work_dir
        seed: random seed
        base_cfg: the base config file
        index: the index of tree, which can be the index of initial tree or its combination
        '''
        options = {'data.train.dataset.ann_file': train_set,
                'data.val.ann_file': test_set,
                'data.test.ann_file': test_set}
        self.scores = None
        self.cfg = Config.fromfile(base_cfg)
        self.cfg.merge_from_dict(options)
        self.model_dir = model_dir
        self.cfg.work_dir = model_dir
        self.cfg.seed = seed
        self.index = index


    def train(self, gpu_id=[0], training_thre=50):
        # gpu_available = available_gpu_exists()
        # if gpu_available == -1:
        #     print('No Space!')
        #     return -1
        # gpu_id = [gpu_available]
        self.cfg.gpu_ids = gpu_id

        if check_available_training(self.cfg.data.train.dataset.ann_file, training_thre):
            # resume from the last result
            if os.path.exists(os.path.join(self.model_dir, 'latest.pth')):
                self.cfg.resume_from = os.path.join(self.model_dir, 'latest.pth')
            # check if the training is finished
            if not os.path.exists(os.path.join(self.model_dir, 'epoch_50.pth')):
                train_one_model(self.cfg)
            print('Training Finished!')
        else:
            print('Training sample is not enough!')
            print()


    def get_scores(self):
        if not os.path.exists(os.path.join(self.model_dir, 'epoch_50.pth')):
            self.scores = 0
        else:
            self.scores = test_one_dataset(self.cfg, os.path.join(self.model_dir, 'epoch_50.pth'))


    def print_counts(self):
        train_annos = self.cfg.data.train.dataset.ann_file
        val_annos = self.cfg.data.val.ann_file

        with open(train_annos,'r') as train_f:
            train_dict = json.load(train_f)

        with open(val_annos,'r') as val_f:
            val_dict = json.load(val_f)

        train_num = len(train_dict['images'])
        val_num = len(val_dict['images'])

        return (train_num, val_num)


    def fusion_test(self, model_list):
        # build the dataloader
        dataset = build_dataset(self.cfg.data.test)
        data_loader = build_dataloader(
            dataset,
            samples_per_gpu=1,
            workers_per_gpu=1,
            dist=False,
            shuffle=False)

        outputs = fusion_inference(model_list, data_loader)

        metric = dataset.evaluate(outputs)
        print(metric)
        if len(metric) == 0:
            return 0
        else:
            return metric['bbox_mAP']


    def decision_test(self, model_list, decision_model):
        # build the dataloader
        dataset = build_dataset(self.cfg.data.test)
        data_loader = build_dataloader(
            dataset,
            samples_per_gpu=1,
            workers_per_gpu=1,
            dist=False,
            shuffle=False)

        outputs = fusion_inference(model_list, data_loader, decision_model=decision_model)

        metric = dataset.evaluate(outputs)
        print(metric)
        if len(metric) == 0:
            return 0
        else:
            return metric['bbox_mAP']


def build_model_list(cfg='configs/bdd/yolox_nano_8x8_300e_base2_bdd.py'):
    # 指定模型的配置文件和 checkpoint 文件路径
    checkpoint_file_list = ['models_bdd_bak/383608746bbcebd51835b33dee4cb5d0/epoch_50.pth',
                            'models_bdd_bak/9cf81d8026a9018052c429cc4e56739b/epoch_50.pth',
                            'models_bdd_bak/0cad859e142b78a748840c72a712722d/epoch_50.pth']

    model_list = []
    for checkpoint_file in checkpoint_file_list:
        # 根据配置文件和 checkpoint 文件构建模型
        model = init_detector(cfg, checkpoint_file, device='cpu')
        model_list.append(model)


def fusion_inference(model_list, data_loader, decision_model=None):

    result_frame_list = []

    for i, data in enumerate(data_loader):

        # print(data['img_metas'][0].data[0][0]['filename'])

        if decision_model is not None:
            # print(data['img'][0].data[0])
            decision_result = inference_model(decision_model, data['img_metas'][0].data[0][0]['filename'])

            print(decision_result['top5'])
            choose_model_list = []
            for model_index in decision_result['top5']:
                choose_model_list.append(model_list[model_index])
        else:
            choose_model_list = model_list

        result_all = [[] for _ in range(2)]
        for model in choose_model_list:
            with torch.no_grad():
                result = model(data['img'][0].data, data['img_metas'][0].data, return_loss=False, rescale=True)
            
            result = result[0]
            
            for class_index in range(2):
                # print(result[class_index].shape)
                for result_each in result[class_index]:
                    result_all[class_index].append(result_each)

        for class_index in range(2):
            result_all[class_index] = np.array(result_all[class_index])

        result_fusion = []
        for result_per_class in result_all:
            if len(result_per_class) > 0:
                boxes = result_per_class[:, :4]
                scores = result_per_class[:, 4]
                dets, _ = nms(boxes, scores, iou_threshold=0.65, score_threshold=0.25)
                # print(dets)
            else:
                dets = np.array([])
            result_fusion.append(dets)
        
        result_frame_list.append(result_fusion)

    return result_frame_list


def tree_fusion(tree1, tree2):
    index1 = tree1.index
    index2 = tree2.index

    index_new = index1 + index2
    index_new.sort()

    json_path_train_new = 'data/bdd_soda_traffic/annotations/sub_bdd/train/'
    index_name_new = ''
    for index_each in index_new:
        index_name_new += index_each + '-'
    index_name_new = index_name_new[:-1]
    print(index_name_new)
    json_path_train_new = json_path_train_new + get_md5(index_name_new) + '.json'
    if not os.path.exists(json_path_train_new):
        json_list = []
        for index_each in index_new:
            json_list.append('data/bdd_soda_traffic/annotations/sub_bdd/train/' + get_md5(index_each) + '.json')
        json_group(json_list, json_path_train_new)

    json_path_val = 'data/bdd_soda_traffic/annotations/sub_bdd/val/'
    json_path_val_new = ''
    for index_each in index_new:
        json_path_val_new = json_path_val_new + index_each + '-'
    json_path_val_new = json_path_val + get_md5(index_name_new) + '.json'
    if not os.path.exists(json_path_val_new):
        json_list = []
        for index_each in index_new:
            json_list.append('data/bdd_soda_traffic/annotations/sub_bdd/val/' + get_md5(index_each) + '.json')
        json_group(json_list, json_path_val_new)

    model_new_dir = os.path.join('/mnt/disk2/models_bdd', get_md5(index_name_new))
    tree_new = Tree(json_path_train_new, json_path_val_new, model_new_dir, index=index_new)
    return tree_new


def inference_model(model, img):
    """Inference image(s) with the classifier.

    Args:
        model (nn.Module): The loaded classifier.
        img (str/ndarray): The image filename or loaded image.

    Returns:
        result (dict): The classification results that contains
            `class_name`, `pred_label` and `pred_score`.
    """
    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    # build the data pipeline
    if isinstance(img, str):
        if cfg.data.test.pipeline[0]['type'] != 'LoadImageFromFile':
            cfg.data.test.pipeline.insert(0, dict(type='LoadImageFromFile'))
        data = dict(img_info=dict(filename=img), img_prefix=None)
    else:
        if cfg.data.test.pipeline[0]['type'] == 'LoadImageFromFile':
            cfg.data.test.pipeline.pop(0)
        data = dict(img=img)
    test_pipeline = Compose(cfg.data.test.pipeline)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]

    # forward the model
    with torch.no_grad():
        scores = model(return_loss=False, **data)
        scores = np.array(scores)
        pred_score = np.max(scores[0])
        pred_label = np.argmax(scores[0])
        print(scores)
        top5_index = heapq.nlargest(5, range(len(scores[0])), scores[0].__getitem__)
        result = {'pred_label': pred_label, 'pred_score': float(pred_score), 'scores': scores[0], 'top5':top5_index}

    return result


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


def dataset_test():
    cityname = 'erfurt'
    cfg = 'configs/bdd/yolox_nano_8x8_300e_base2.py'
    tree = Tree(train_set='data/cityscapes/json/train/'+cityname+'.json', test_set='data/cityscapes/json/val/'+cityname+'.json', model_dir='models/'+cityname, base_cfg=cfg)
    # tree.train(gpu_id=[0])
    # tree.get_scores()
    # weight_list = ['/mnt/disk2/models_bdd/b3fc0f73e6dd04b6a8406754ef7a73bb/epoch_50.pth', '/mnt/disk2/models_bdd/a40f9f4f6e4db0f95d99703ff46c2fe9/epoch_50.pth', '/mnt/disk2/models_bdd/47b3e2ef756d2119126cb0fe9210d079/epoch_50.pth', '/mnt/disk2/models_bdd/4ac84eb014d2aa596bd6daa2bcb5d39c/epoch_50.pth', '/mnt/disk2/models_bdd/ec2650cc35fea239c3c99273d993b1c4/epoch_50.pth', '/mnt/disk2/models_bdd/96d83bb191c923221937ad27f07a90ea/epoch_50.pth', '/mnt/disk2/models_bdd/a66126f5dc69e6ff8db56a27ee2c2cfe/epoch_50.pth', '/mnt/disk2/models_bdd/fe2735666087519d72dc3fc1e2eb930e/epoch_50.pth', '/mnt/disk2/models_bdd/cfb796f34e04c8e5b908ffc3ac201cd8/epoch_50.pth', '/mnt/disk2/models_bdd/8b5306e3d771e16ad901c1fb4ebe4123/epoch_50.pth']

    weight_list = list_md5_weight()

    model_list = []
    for checkpoint_file in weight_list:
        # 根据配置文件和 checkpoint 文件构建模型
        model = init_detector(cfg, checkpoint_file, device='cpu')
        model_list.append(model)

    model_x = init_detector('configs/bdd/yolox_x_8x2_300e_all3.py', 'all_model/epoch_50.pth', device='cpu')

    # decision model
    config_file = '/home/liyunzhe/mmclassification/work_dirs/choose_net_run_cross/choose_net_run_cross.py'
    checkpoint_file = '/home/liyunzhe/mmclassification/work_dirs/choose_net_run_cross/epoch_100.pth'  # this is the 7w5 model
    # checkpoint_file = '/home/liyunzhe/mmclassification/work_dirs/220221-best-11-incre/latest.pth'
    device = 'cuda:1'

    decision_model = init_model(config_file, checkpoint_file, device=device)

    result_all_n = tree.fusion_test([model_list[9]])
    result_decision = tree.decision_test(model_list, decision_model)
    result_all_x = tree.fusion_test([model_x])
    print()
    print('result_all_nano', result_all_n)
    print('result_decision', result_decision)
    print('result_all_x', result_all_x)


if __name__ == '__main__':
    dataset_test()
