# Coding: UTF-8 
# Created by 11 at 2021/1/14
# This "mSplit_ImageDataSet.py" will implement function about: 划分训练集与测试集

import os
import random
from shutil import copy2

root_path = './data/mDataset/'
data_source_path = './data/Flowers/'
train_path = './data/mDataset/train/'
test_path = './data/mDataset/test/'


def give_root_dir(data_path):
    if not os.path.exists(data_path):
        os.makedirs(data_path)
    else:
        print(f'错误，路径{data_path}对应的文件夹已存在')
    dic = ['train', 'test']
    for i in range(0, 2):
        current_path = data_path + dic[i] + '/'
        isExists = os.path.exists(current_path)
        if not isExists:
            os.makedirs(current_path)
            print('已生成' + dic[i] + '文件夹')
        else:
            print(f'错误，路径{current_path}对应的文件夹已存在')
    return


def get_image_classes(source_path):
    classes_name_list = os.listdir(source_path)
    classes_num = len(classes_name_list)
    return classes_name_list, classes_num


def give_leaf_dir(source_path, change_path):
    classes_name_list, classes_num = get_image_classes(source_path)
    for i in range(0, classes_num):
        current_class_path = os.path.join(change_path, classes_name_list[i])
        isExists = os.path.exists(current_class_path)
        if not isExists:
            os.makedirs(current_class_path)
            print('已生成 ' + classes_name_list[i] + '文件夹')
        else:
            print(f'错误，路径{current_class_path}对应的文件夹已存在')


def give_train_test_data(source_path, train_path, test_path):
    classes_name_list, classes_num = get_image_classes(source_path)
    give_leaf_dir(source_path, train_path)
    give_leaf_dir(source_path, test_path)

    for i in range(0, classes_num):
        source_image_dir = os.listdir(source_path + classes_name_list[i] + '/')
        random.shuffle(source_image_dir)
        train_image_list = source_image_dir[0:int(0.7 * len(source_image_dir))]
        test_image_list = source_image_dir[int(0.7 * len(source_image_dir)):]

        for train_image in train_image_list:
            origins_train_image_path = source_path + classes_name_list[i] + '/' + train_image
            new_train_image_path = train_path + classes_name_list[i] + '/'
            copy2(origins_train_image_path, new_train_image_path)
        for test_image in test_image_list:
            origins_test_image_path = source_path + classes_name_list[i] + '/' + test_image
            new_test_image_path = test_path + classes_name_list[i] + '/'
            copy2(origins_test_image_path, new_test_image_path)


if __name__ == '__main__':
    give_root_dir(root_path)
    give_train_test_data(data_source_path, train_path, test_path)
