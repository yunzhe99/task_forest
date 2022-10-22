from mmdet.apis import init_detector, inference_detector, async_inference_detector
import mmcv
import time
from tqdm import tqdm
from mmcv import Config
import matplotlib.pyplot as plt
from mmcv.ops.nms import nms
import numpy as np
import asyncio
import torch
from mmdet.utils.contextmanagers import concurrent
from mmdet.datasets import (build_dataloader, build_dataset)


def batch_test():
    # 指定模型的配置文件和 checkpoint 文件路径
    config_file = 'configs/bdd/yolox_nano_8x8_300e_base2_bdd.py'
    checkpoint_file = 'models_bdd/383608746bbcebd51835b33dee4cb5d0/epoch_50.pth'

    # config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
    # checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'

    # 根据配置文件和 checkpoint 文件构建模型
    model = init_detector(config_file, checkpoint_file, device='cuda:1')

    time_consumed_list = []
    for batch_size in tqdm(range(1, 30)):
        # 测试单张图片并展示结果
        # batch_size = 1
        img = ['demo/demo.jpg' for _ in range(batch_size)]

        time_start = time.time()

        result = inference_detector(model, img)

        time_end = time.time()

        time_consumed = (time_end-time_start) * 1000
        print(time_consumed/batch_size)

        # # 或者将可视化结果保存为图片
        # model.show_result(img, result, out_file='result.jpg')

        time_consumed_list.append(time_consumed/batch_size)

    print()
    print(np.mean(time_consumed_list[1:]))

    time_consumed_list = time_consumed_list[1:]

    index = [i+1 for i in range(len(time_consumed_list))]
    plt.plot(index, time_consumed_list)
    plt.xlabel('Batch Size')
    plt.ylabel('Average Latency for 1 frame')
    plt.title('Batch Size vs Latecny')

    plt.savefig('figures/latency.png')


def single_test_frame():
    # 指定模型的配置文件和 checkpoint 文件路径
    config_file = 'configs/bdd/yolox_nano_8x8_300e_base2_bdd.py'
    checkpoint_file = 'models_bdd/383608746bbcebd51835b33dee4cb5d0/epoch_50.pth'

    cfg = Config.fromfile(config_file)

    cfg.model.test_cfg.nms = None

    # print(f'Config:\n{cfg.pretty_text}')

    # 根据配置文件和 checkpoint 文件构建模型
    model = init_detector(cfg, checkpoint_file, device='cuda:1')

    img = 'demo/demo.jpg'

    result = inference_detector(model, img)

    for result_per_class in result:
        boxes = result_per_class[:, :4]
        scores = result_per_class[:, 4]
        dets, _ = nms(boxes, scores, iou_threshold=0.65)
        print(dets.shape)

    # model.show_result(img, result, out_file='result.jpg')


def fusion_test_frame():
    # 指定模型的配置文件和 checkpoint 文件路径
    config_file = 'configs/bdd/yolox_nano_8x8_300e_base2_bdd.py'
    checkpoint_file_list = ['models_bdd_bak/383608746bbcebd51835b33dee4cb5d0/epoch_50.pth',
                            'models_bdd_bak/9cf81d8026a9018052c429cc4e56739b/epoch_50.pth',
                            'models_bdd_bak/0cad859e142b78a748840c72a712722d/epoch_50.pth']

    cfg = Config.fromfile(config_file)

    # cfg.model.test_cfg.nms = None

    # print(f'Config:\n{cfg.pretty_text}')
    model_list = []
    for checkpoint_file in checkpoint_file_list:
        # 根据配置文件和 checkpoint 文件构建模型
        model = init_detector(cfg, checkpoint_file, device='cuda:1')
        model_list.append(model)

    img = 'demo/demo.jpg'

    result_all = [[] for _ in range(2)]
    for model in model_list:
        result = inference_detector(model, img)
        for class_index in range(2):
            # print(result[class_index].shape)
            for result_each in result[class_index]:
                result_all[class_index].append(result_each)

    for class_index in range(2):
        result_all[class_index] = np.array(result_all[class_index])

    # print(result_all[1].shape)

    result_fusion = []
    for result_per_class in result_all:
        boxes = result_per_class[:, :4]
        scores = result_per_class[:, 4]
        dets, _ = nms(boxes, scores, iou_threshold=0.65, score_threshold=0.25)
        print(dets)
        result_fusion.append(dets)

    model.show_result(img, result_fusion, out_file='result.jpg')


async def detect():
    config_file = 'configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py'
    checkpoint_file = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
    device = 'cuda:1'
    model = init_detector(config_file, checkpoint=checkpoint_file, device=device)

    # 此队列用于并行推理多张图像
    streamqueue = asyncio.Queue()
    # 队列大小定义了并行的数量
    streamqueue_size = 10

    for _ in range(streamqueue_size):
        streamqueue.put_nowait(torch.cuda.Stream(device=device))

    # # 测试单张图片并展示结果
    # img = ['demo/demo.jpg', 'demo/demo.jpg']  # or 或者 img = mmcv.imread(img)，这样图片仅会被读一次

    time_consumed_list = []

    for batch_size in tqdm(range(1, 1000)):
        # 测试单张图片并展示结果
        img = mmcv.imread('demo/demo.jpg')

        time_start = time.time()

        async with concurrent(streamqueue):
            result = await async_inference_detector(model, img)

        time_end = time.time()

        time_consumed = (time_end-time_start) * 1000

        print(time_consumed)

        time_consumed_list.append(time_consumed)

    print()
    print(np.mean(time_consumed_list[1:]))

    # # 或者将可视化结果保存为图片
    # model.show_result(img, result, out_file='result.jpg')


def parallel_detect():
    asyncio.run(detect())


def fusion_inference(model_list, data_loader):
    for i, data in enumerate(data_loader):
        print(type(data))


def test_with_cfg(cfg):
    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=1,
        dist=False,
        shuffle=False)

    # 指定模型的配置文件和 checkpoint 文件路径
    checkpoint_file_list = ['models_bdd_bak/383608746bbcebd51835b33dee4cb5d0/epoch_50.pth',
                            'models_bdd_bak/9cf81d8026a9018052c429cc4e56739b/epoch_50.pth',
                            'models_bdd_bak/0cad859e142b78a748840c72a712722d/epoch_50.pth']

    model_list = []
    for checkpoint_file in checkpoint_file_list:
        # 根据配置文件和 checkpoint 文件构建模型
        model = init_detector(cfg, checkpoint_file, device='cuda:1')
        model_list.append(model)

    outputs = fusion_inference(model_list, data_loader)


if __name__ == '__main__':
    # parallel_detect()
    # batch_test()
    # config_file = 'configs/bdd/yolox_nano_8x8_300e_base2_bdd.py'
    
    # test_with_cfg()

    fusion_test_frame()
