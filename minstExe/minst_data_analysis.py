# coding=utf-8
'''
minst_data_analysis.py
对minst数据集进行数据分析和数据处理和介绍coco模块
https://github.com/DataXujing/EfficientDet_pytorch(值得学习的入门项目)
数据显示：
（1）loss显示
（2）目标检测显示（将框画出来）

数据分析：
（1）分析数据集的总大小；每个类别的多少；是否有类别不平衡问题（cls_analysis）
（2）分析图像的最大最小分辨率（statistical_img_size）

数据处理：
（1）将总数据集随机划分成训练集和验证集（split_trainval_to_train_val）
（2）分别生成训练集和验证集的标注，为数据加载和训练模型做准备（gen_data_anns）
（3）构造自己的数据加载集，

数据增强：

COCO数据集分类：
共有物体检测 (Detection)、
人体关键点检测 (Keypoints)、
图像分割 (Stuff)、
图像描述生成 (Captions) 四个类别的比赛任务

coco加载集使用：：
dset.CocoCaptions返回的格式：
coco:{
    anns:所有的标注
    catToimgs:类别为key,value是所有的img_id
    cats:所有类别
    dataset:原始数据，包括images,annotations,categories
    imgToAnns:每张图片对应的ann,一对多
    imgs:图片信息
}
ids:对应的index
root:图片根目录

作者：Jay
创建时间：2020-07-01
'''
import glob
import numpy as np
import sys
import cv2
from tqdm import tqdm
import os
import json
import collections
import yaml
from PIL import Image
import matplotlib.pyplot as plt  # plt 用于显示图片
import matplotlib.image as mpimg  # mpimg 用于读取图片


# 数据显示
def data_show():
    path = 'E:\\temp\\train_dataset_part1\\'
    img_pth = glob.glob(path + 'image/*')
    for i in img_pth:
        imgs = glob.glob(i + '/*.jpg')
        for img in imgs:
            image = cv2.imread(img)
            ann_pth = img.replace('image', 'image_annotation')
            ann_pth = ann_pth.replace('jpg', 'json')
            print(ann_pth)
            with open(ann_pth, encoding='utf-8') as fa:
                anns = json.load(fa)
                annotations = anns['annotations']
            for ann in annotations:
                xmin, ymin, xmax, ymax = ann['box']
                xmin, ymin, xmax, ymax = int(xmin), int(ymin), int(xmax), int(ymax)
                # 传入矩形的顶点
                cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 0, 255))
            cv2.imwrite('result.png', image)
            break
        break
    print("生成图片存在已经生成")


# 数据集加载
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as det
import torchvision.transforms as transforms

# print('[Epoch:%d, Iter(img):%d, Pos_pair_num:%d, Neg_pair_num:%d] loss: %.4f, time(s): %d' %
#       (epoch + 1, i, pos_pair_num, neg_pair_num, running_loss / (pos_pair_num + neg_pair_num), end_time - start_time))
train_path = '../data/MNIST/train/'
test_path = '../data/MNIST/test/'


def save_json(images, annotations, save_path):
    ann = {}  # coco类型标签
    ann['type'] = 'instances'
    ann['images'] = images
    ann['annotations'] = annotations
    category = [
        {'id': 1, 'name': '短外套', 'supercategory': 'class1'},
        {'id': 2, 'name': '古风', 'supercategory': 'class2'},
        {'id': 3, 'name': '短裤', 'supercategory': 'class3'},
        {'id': 4, 'name': '短袖上衣', 'supercategory': 'class4'},
        {'id': 5, 'name': '长半身裙', 'supercategory': 'class5'},
        {'id': 6, 'name': '背带裤', 'supercategory': 'class6'},
        {'id': 7, 'name': '长袖上衣', 'supercategory': 'class7'},
        {'id': 8, 'name': '长袖连衣裙', 'supercategory': 'class8'},
        {'id': 9, 'name': '短马甲', 'supercategory': 'class9'},
        {'id': 10, 'name': '短裙', 'supercategory': 'class10'},
        {'id': 11, 'name': '背心上衣', 'supercategory': 'class11'},
        {'id': 12, 'name': '短袖连衣裙', 'supercategory': 'class12'},
        {'id': 13, 'name': '长袖衬衫', 'supercategory': 'class13'},
        {'id': 14, 'name': '中等半身裙', 'supercategory': 'class14'},
        {'id': 15, 'name': '无袖上衣', 'supercategory': 'class15'},
        {'id': 16, 'name': '长外套', 'supercategory': 'class16'},
        {'id': 17, 'name': '无袖连衣裙', 'supercategory': 'class17'},
        {'id': 18, 'name': '连体衣', 'supercategory': 'class18'},
        {'id': 19, 'name': '长马甲', 'supercategory': 'class19'},
        {'id': 20, 'name': '长裤', 'supercategory': 'class20'},
        {'id': 21, 'name': '吊带上衣', 'supercategory': 'class21'},
        {'id': 22, 'name': '中裤', 'supercategory': 'class22'},
        {'id': 23, 'name': '短袖衬衫', 'supercategory': 'class23'},
    ]
    ann['categories'] = category
    json.dump(ann, open(save_path, 'w'))


# 23类类别信息（其中'古风'与'古装'同为第2类）
CLASS_DICT = collections.OrderedDict({
    '短外套': 1,
    '古风': 2, '古装': 2,
    '短裤': 3,
    '短袖上衣': 4,
    '长半身裙': 5,
    '背带裤': 6,
    '长袖上衣': 7,
    '长袖连衣裙': 8,
    '短马甲': 9,
    '短裙': 10,
    '背心上衣': 11,
    '短袖连衣裙': 12,
    '长袖衬衫': 13,
    '中等半身裙': 14,
    '无袖上衣': 15,
    '长外套': 16,
    '无袖连衣裙': 17,
    '连体衣': 18,
    '长马甲': 19,
    '长裤': 20,
    '吊带上衣': 21,
    '中裤': 22,
    '短袖衬衫': 23})


# （1）划分训练集和测试集的时候按照每个类别分类
# 训练集共有10个类别，每个类别大概6000张图片，没有类别不平衡问题
# 训练集共有10个类别，每个类别大概1000张图片，没有严重类别不平衡问题
def cls_analysis(path):
    '''有多少类别，每个类别的数据频次,正常来说应该是从标签分析；但本数据集图片所在目录即标签'''
    dirs = glob.glob(path + '*')
    cls_total_num = len(dirs)
    cls_each_num = np.zeros(cls_total_num)
    for index, cls in enumerate(dirs):
        # print(cls)
        imgs = glob.glob(cls + '/*.png')
        cls_each_num[index] = len(imgs)
    return cls_total_num, cls_each_num


# 将train划分成训练集和验证集(只分名字，标注之后再说)
# 之后再根据名字去过去标签
def split_trainval_to_train_val(path, ratio, mode='class'):
    '''
        分类：(class)
        假设路径结构：root/images/class_name/jpg path
        若不是，先整理成这种结构
        实例分割：(instance)
        假设如路径结构：root/images/instance_name/jpg path
        则遍历的时候再统计类别
    '''
    train = []
    val = []
    if mode == 'class':
        cls_each = glob.glob(path + '*')
        for cls in cls_each:
            cls_name = cls[cls.rfind('\\') + 1:]
            imgs_name_list = os.listdir(cls)
            imgs_name_list = [cls_name + '/' + img_name for img_name in imgs_name_list]
            cls_len = len(imgs_name_list)
            # print(cls_len)
            len_train, len_val = int(round(cls_len * ratio)), int(round(cls_len * (1 - ratio)))
            # print(len_train,len_val)
            cls_train, cls_val = torch.utils.data.random_split(imgs_name_list, [len_train, len_val])
            train.extend(cls_train)
            val.extend(cls_val)
    else:  # instance           #暂时不知道一个图片中又多个类别bbox时，应该如何分类，以第一个为类别
        train = []
        val = []
        trainval = {}
        path = path + '/image_annotation/'
        each_instance = glob.glob(path + '*')
        for instance in tqdm(each_instance):
            # print(instance)
            anns_pth = glob.glob(instance + '\*.json')
            for ann_pth in anns_pth:
                with open(ann_pth, encoding='utf-8') as fa:
                    anns = json.load(fa)['annotations']
                    ann_pth_list = ann_pth.split('\\')  # windwos/linux不同
                    for ann in anns:
                        label = ann['label']
                        if trainval.get(label) == None:
                            trainval[label] = [
                                'image/' + ann_pth_list[-2] + '/' + ann_pth_list[-1].replace('json', 'jpg')]
                        else:
                            trainval[label].append(
                                'image/' + ann_pth_list[-2] + '/' + ann_pth_list[-1].replace('json', 'jpg'))
                        break
        for cls_name, imgs_name in tqdm(trainval.items()):
            cls_len = len(imgs_name)
            # print(cls_len)     #该数据集有比较大的数据不平衡问题
            len_train, len_val = int(round(cls_len * ratio)), int(round(cls_len * (1 - ratio)))
            cls_train, cls_val = torch.utils.data.random_split(imgs_name, [len_train, len_val])
            train.extend(cls_train)
            val.extend(cls_val)
    json.dump(train, open('../gen/train_raw.json', encoding='utf-8', mode='w'))
    json.dump(val, open('../gen/val_raw.json', encoding='utf-8', mode='w'))
    return train, val


# 生成数据集标签,coco类型，images,annotations,先调用split_trainval_to_train_val生成train_raw,val_raw
def gen_data_anns(path, train_pth, val_pth, mode='class', bbox_mode='coco'):  # coco和voc生成的标签不一样
    if mode == 'class':
        anns_train = []  # file_name,lable
        anns_val = []
        # 生成train标签
        for img_pth in train_pth:
            ann = {}
            # print(img_pth)
            ann['file_name'] = img_pth
            ann['label'] = img_pth[0:img_pth.find('/')]
            anns_train.append(ann)
        # 生成val标签
        for img_pth in val_pth:
            ann = {}
            ann['file_name'] = img_pth
            ann['label'] = img_pth[0:img_pth.find('/')]
            anns_val.append(ann)
        json.dump(anns_train, open('./gen/train.json', encoding='utf-8', mode='w'))
        json.dump(anns_val, open('./gen/val.json', encoding='utf-8', mode='w'))
    else:
        img_id = 1
        ann_id = 0
        for index, pth in enumerate([train_pth, val_pth]):
            imgs = []
            anns = []
            for img_pth in tqdm(pth):
                img_name = path + img_pth
                # print(img_name)
                # 获取图片信息
                img = cv2.imread(img_name)
                h, w, _ = img.shape
                del img
                imgs.append({'file_name': img_pth,
                             'id': img_id,
                             'height': h,
                             'width': w})
                # 获取标注信息
                ann_name = img_name.replace('image', 'image_annotation')
                ann_name = ann_name.replace('jpg', 'json')
                with open(ann_name, encoding='utf-8') as fa:
                    img_anns = json.load(fa)
                for ann in img_anns['annotations']:
                    instance_id = ann['instance_id']
                    xmin = float(ann['box'][0])
                    ymin = float(ann['box'][1])
                    box_w = float(ann['box'][2] - ann['box'][0] + 1)
                    box_h = float(ann['box'][3] - ann['box'][1] + 1)
                    cls_id = CLASS_DICT[ann['label']]
                    anns.append({
                        'id': ann_id,  # 一定要是按照序列的
                        'image_id': img_id,
                        'bbox': [xmin, ymin, box_w, box_h],  # 对bbox的处理要注意，不同地方处理不同
                        'category_id': cls_id,
                        'instance_id': instance_id,
                        'segmentation': [],
                        "area": box_w * box_h,
                        'iscrowd': 0
                    })
                    ann_id = ann_id + 1
                img_id = img_id + 1
                if img_id > 50:
                    break
            if index == 0:
                save_json(imgs, anns, '../gen/train_instance.json')
            else:
                save_json(imgs, anns, '../gen/val_instance.json')
        del imgs
        del anns


# 数据加载+数据增强
def coco_dataset():
    configs = yaml.safe_load(open('../config/coco.yml').read())
    train_augmentation = transforms.Compose([  # transforms.RandomResizedCrop(800),#随机对图像进行裁剪，图像的标签会变化
        # transforms.Resize(800),  #只会根据短边修改到800，长边会变长
        transforms.RandomCrop(800, padding=1, pad_if_needed=True),  # 不够800填充黑色
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomVerticalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize(configs['mean'], configs['std'])])
    cap = det.CocoDetection(root='E:\\temp\\train_dataset_part1\\',
                            annFile='../gen/val_instance.json',
                            transform=train_augmentation)  # 注意没有s
    print('Number of sampler {}'.format(len(cap)))
    for each in cap:
        img, target = each  ## load 4th sample
        # print(img.size())  # 转化后图像
        # print(target)  # 标注,图像改变了大小，但是标注却没有？？？（不是学错了）
    train_db, val_db = torch.utils.data.random_split(cap, [7, 3])  # 使用该方法会按照类别分类（放心使用）
    train_loader = DataLoader(train_db, batch_size=5, shuffle=True)
    val_loader = DataLoader(val_db, batch_size=1, shuffle=True)
    for batch_index, data in enumerate(train_loader):
        # data[0]存储的是原始img数据，len(data[0])=batch_size
        # data[1]存储时原始标注信息
        print(batch_index, len(data[0]), len(data[1]))
        print(len(data[1]))

        break
    # for batch_index,data in enumerate(val_loader):
    #     print(batch_index,data)


class MyData(Dataset):
    '''https://blog.csdn.net/yizhishuixiong/article/details/105224710(加载自己的数据集)'''

    def __init__(self, root, ann_file, transform=None, target_transform=None):
        '''初始化读取数据集'''
        super(MyData, self).__init__()
        pass

    def __getitem__(self, id):
        '''读取指定id，并且返回该数据'''
        pass

    def __len__(self):
        '''获取数据集的总大小'''
        pass


# （2）图像的大小，max,min,这里貌似没有bbox标注
# 分辨率都是[28, 28, 28, 28]       img_min_height,img_max_height,img_min_width,img_max_width
def statistical_img_size(path):
    img_max_height = 0
    img_min_height = sys.maxsize
    img_max_width = 0
    img_min_width = sys.maxsize
    cls_each = glob.glob(path + '*')  # glob的可以是相对路径
    for cls in tqdm(cls_each):
        # print(cls)
        imgs_name = glob.glob(cls + '/*.png')
        for img_name in imgs_name:
            img = cv2.imread(img_name)  # 读取灰色图
            height = img.shape[0]
            width = img.shape[1]
            if height < img_min_height:
                img_min_height = height
            else:
                if height > img_max_height:
                    img_max_height = height
            if width < img_min_width:
                img_min_width = width
            else:
                if width > img_max_width:
                    img_max_width = width
    return [img_min_height, img_max_height, img_min_width, img_max_width]


if __name__ == '__main__':
    data_show()
