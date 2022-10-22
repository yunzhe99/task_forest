import os
import sh
import copy
import json
import pynvml
import hashlib

import pandas as pd

from mmcv import Config


def load_cfg(config_file='configs/bdd/yolox_nano_8x8_300e_all3.py'):
    cfg = Config.fromfile(config_file)

    options = {'data.train.dataset.ann_file': 'test',
                'data.val.ann_file': 'test',
                'data.test.ann_file': 'test'}
    cfg.merge_from_dict(options)
    cfg.work_dir = 'test'
    cfg.gpu_ids = 0
    print(f'Config:\n{cfg.pretty_text}')


def available_gpu_exists(memory_needed=4500):
    no_memory_flag = -1
    pynvml.nvmlInit()
    for gpu_index in range(2):
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        if meminfo.free / 1024 / 1024 > memory_needed:
            return gpu_index
    return no_memory_flag


def task_name_get(task_data='data/cityscapes/json/train'):
    # get the list of task name from a dir
    file_list = os.listdir(task_data)
    file_name_list = []
    for file_name in file_list:
        # if file_name[-6] == 'n':  # the train and val set need to be grouped
        #     file_name_list.append(file_name[:-11])
        file_name_list.append(file_name[:-5])
        # print(file_name[:-5])
    return file_name_list


def copy_images(dir_path='data/cityscapes/leftImg8bit/train'):
    city_name_list = os.listdir(dir_path)
    for city_name in city_name_list:
        image_list = os.listdir(os.path.join('data/cityscapes/leftImg8bit/train', city_name))
        for image_path in image_list:
            sh.cp(os.path.join(os.path.join('data/cityscapes/leftImg8bit/train', city_name), image_path), 'data/cityscapes/images')


def json_add(json1, json2):
    
    json_dict_new = copy.deepcopy(json1)
    
    image_id_shift = image_id = len(json_dict_new['images'])
    anno_id = len(json_dict_new['annotations'])

    for item_each in json2['images']:
        image_id = len(json_dict_new['images'])
        item_each['id'] = image_id
        image_id += 1
        # print(item_each)
        json_dict_new['images'].append(item_each)
    
    for item_each in json2['annotations']:
        item_each['id'] = anno_id
        anno_id += 1
        item_each['image_id'] += image_id_shift
        # print(item_each)
        json_dict_new['annotations'].append(item_each)
    
    return json_dict_new


def get_md5(raw_str):
    md5 = hashlib.md5()
    md5.update(raw_str.encode('utf-8'))
    return md5.hexdigest()


def json_group(json_list, save_path):
    json_dict_list = []
    for json_each in json_list:
        with open(json_each,'r') as load_f:
            load_dict = json.load(load_f)
            json_dict_list.append(load_dict)

    json_new = json_dict_list[0]  # initial

    for json_index in range(1, len(json_dict_list)):
        json_file = json_dict_list[json_index]
        json_new = json_add(json_new, json_file)
    
    with open(save_path, 'w') as f:
        json.dump(json_new, f)
        print('save the grouped json file to', save_path)


def check_available_training(training_json, training_thre):
    with open(training_json,'r') as load_f:
        load_dict = json.load(load_f)
    
    image_length = len(load_dict['images'])

    if image_length > training_thre:
        return True
    else:
        return False


def text2md5_json(dir_path='data/bdd_soda_traffic/annotations/sub_bdd/train'):
    file_name_list = os.listdir(dir_path)
    for file_name in file_name_list:
        file_name_pure, file_ext = os.path.splitext(file_name)
        print(file_name_pure)
        md5_name = get_md5(file_name_pure)
        print(md5_name)
        print(md5_name+file_ext)
        print()
        sh.mv(os.path.join(dir_path, file_name), os.path.join(dir_path, md5_name+file_ext))


def text2md5_dir(dir_path='/mnt/disk2/models_bdd'):
    dir_name_list = os.listdir(dir_path)
    for dir_name in dir_name_list:
        print(dir_name)
        md5_name = get_md5(dir_name)
        print(md5_name)
        print()
        sh.mv(os.path.join(dir_path, dir_name), os.path.join(dir_path, md5_name))


def json_show(json_path='data/bdd_soda_traffic/annotations/sub_bdd/train/383608746bbcebd51835b33dee4cb5d0.json'):
    with open(json_path,'r') as load_f:
        load_dict = json.load(load_f)

    # for key in load_dict.keys():
    #     print(load_dict[key])

    # for annotation in load_dict['annotations'][:1000]:
    #     print(annotation['id'])
    
    # print(load_dict['images'])

    for image in load_dict['images'][:1000]:
        print(os.path.join('/home/liyunzhe/mmdetection/data/bdd_soda_traffic/train/images', image['file_name']))
    

def json_split_each(json_path, target_dir='/mnt/disk2/detection_ped_car/json/val'):
    with open(json_path,'r') as load_f:
        load_dict = json.load(load_f)

    print(load_dict.keys())

    for image in load_dict['images']:
        dataset = {'categories': load_dict['categories'], 'annotations': [], 'images': [image]}
        image_id = image['id']

        for annotation in load_dict['annotations']:
            if annotation['image_id'] == image_id:
                dataset['annotations'].append(annotation)
        
        if not exists(os.path.join(target_dir, os.path.splitext(os.path.split(json_path)[1])[0])):
            os.mkdir(os.path.join(target_dir, os.path.splitext(os.path.split(json_path)[1])[0]))

        save_path = os.path.join(target_dir, os.path.splitext(os.path.split(json_path)[1])[0], str(image_id)+'.json')

        with open(save_path, 'w') as f:
            json.dump(dataset, f)
            print('save the grouped json file to', save_path)


def best_count(txt_path='forest/model_log/initial_models_bdd_train.txt'):
    label_list = []
    with open(txt_path, "r") as f:
        for line in f.readlines():
            line = line.strip('\n')  #去掉列表中每一个元素的换行符
            label_list.append(line.split()[1])
    result = pd.value_counts(label_list)
    print(result)


if __name__ == '__main__':
    # json_split_each('data/bdd_soda_traffic/annotations/bdd_val.json')
    best_count(txt_path='forest/model_log/selected_models_bdd_train.txt')
