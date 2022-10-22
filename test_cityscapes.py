import os
from tools import get_md5
from tree import Tree


def test_one_tree(index, checkpoint_file_list):
    train_set = os.path.join('data/cityscapes/json/train', index[0]+'.json')
    test_set = os.path.join('data/cityscapes/json/val', index[0]+'.json')
    model_dir = os.path.join('models_cityscapes', index[0])

    tree = Tree(train_set, test_set, model_dir, index=index, base_cfg='configs/bdd/yolox_nano_8x8_300e_base2.py')

    tree.fusion_test(checkpoint_file_list)


def test_one_frame(test_json_path, checkpoint_file_list):
    tree = Tree(test_set=test_json_path, base_cfg='configs/bdd/yolox_nano_8x8_300e_base2.py')
    tree.fusion_test(checkpoint_file_list)


if __name__ == '__main__':

    task_index_list = ['000-010-020-100-110-120-200-210-210-220-220-300-310-320-400-410-420-500-510-520-600-610-620-630',
    '002-012-022-112-122-202-212-222-302-312-322-402-412-422-502-512-522-602-612-622-632',
    '003-013-023-213-313-413-503-513-523-613-633',
    '004-014-024-034-114-204-214-224-304-314-324-404-414-424-504-514-524-604-614-624-634',
    '010-012-013-014-016',
    '000-002-003-004-006-200-202-204-210-211-212-213-214-216-300-302-304-310-312-313-314',
    '020-021-022-023-024-025-026-220-222-224-320-322-324',
    '100-110-112-114-400-402-404-410-411-412-413-414-415-500-502-503-504-510-512-513-514',
    '120-122-420-422-424-426-520-522-523-524-526'
    ]

    checkpoint_file_list = []

    for task_index in task_index_list:
        checkpoint_file_list.append(os.path.join('/mnt/disk2/models_bdd', get_md5(task_index), 'epoch_50.pth'))

    # test_one_tree(index=['aachen'], checkpoint_file_list=checkpoint_file_list[:1])
    test_one_frame(test_json_path='/mnt/disk2/cityscapes/test/aachen/113.json', checkpoint_file_list=checkpoint_file_list)


    