#coding=utf-8

'''
https://www.cnblogs.com/wildgoose/p/12905004.html(教程：voc数据集的完整训练过程)

定义数据集及数据增强
定义模型
模型训练
模型验证
可视化

faster-rcnn简介：（把vgg改造成分类和回归网络）
1.RPN模块（参数：rpn_fg_iou_thresh，rpn_bg_iou_thresh，rpn_nms_thresh）
（1）RPN只负责区域生成， 保证recall， 而没必要细分每一个区域属于哪一个类别， 因此只需
要前景与背景两个类别， 前景即有物体， 背景则没有物体
（2）RPN通过计算Anchor与标签的IoU来判断一个Anchor是属于前景还是背景（预测值），同时需要预测物体的位置，即预测物体偏移
（3）根据标签bbox和预测bbox的中心点，长、宽，计算偏移（真实值）
2.Roi-Pooling模块
3.RCNN模块
'''
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

import utils
import yaml
from tqdm import tqdm
from engine import train_one_epoch, evaluate
import numpy as np
import argparse

def get_parser():
    parser=argparse.ArgumentParser(description='my first detecion model')
    parser.add_argument('--checkpoint',required=False)
    return parser

#配置加载
configs=yaml.load(open('./config/coco.yml').read())
#数据处理
trans=transforms.Compose([transforms.RandomCrop(800,1,True,0),
                         transforms.RandomVerticalFlip(0.5),
                         transforms.RandomHorizontalFlip(0.5),
                         transforms.ToTensor(),
                         transforms.Normalize(configs['mean'],configs['std'])])
coco_dataset=datasets.CocoDetection(root='E:\\temp\\train_dataset_part1\\',
                               annFile='./gen/train_instance.json',
                               transform=trans)

ratio=configs['train_val_ratio']
train_num=int(round(len(coco_dataset)*ratio))
val_num=int(round(len(coco_dataset)*(1-ratio)))
train_set,val_set=torch.utils.data.random_split(coco_dataset,[train_num,val_num])

#collate_fn的作用时将img和img放在一起，ann和ann放在一起
train_dataloader=DataLoader(train_set,batch_size=configs['batch_size'],shuffle=True,collate_fn=utils.collate_fn)
val_dataloader=DataLoader(val_set,batch_size=configs['batch_size'],shuffle=True,collate_fn=utils.collate_fn)

#模型创建
#https://blog.csdn.net/u010361236/article/details/90045350
#coco_model=models.vgg16(num_classes=23)   #可以配置vgg16的参数个数,只能用来分类
#https://blog.csdn.net/u013685264/article/details/100564660(faster_rcnn根据自己需要的修改)
#https://pytorch.org/tutorials/intermediate/quantized_transfer_learning_tutorial.html(torch官网讲解)
coco_model = models.detection.fasterrcnn_resnet50_fpn(pretrained=False,pretrained_backbone=False) #func，加载voc类型数据集
#自己加载faster rcnn
faster_rcnn_pretrained_model=torch.load('model/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth')
coco_model.load_state_dict(faster_rcnn_pretrained_model)
# replace the classifier with a new one, that has num_classes which is user-defined
num_classes = 24  # 1 class (person) + background
# get number of input features for the classifier
in_features = coco_model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
coco_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

#模型优化
loss_func=nn.CrossEntropyLoss()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# construct an optimizer
params = [p for p in coco_model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=configs['learning_rate'],
                            momentum=0.9, weight_decay=0.0005)
# 当你想恢复某一阶段的训练（或者进行测试）时，那么就可以读取之前保存的网络模型参数等。
parser=get_parser()
args=parser.parse_args()
start_epoch=0
if args.checkpoint != None:
    checkpoint = torch.load('minst.pth')
    coco_model.load_state_dict(checkpoint['model'])    #此时就将coco_model加载到内存中
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch'] + 1
# lr_scheduler每3个epoch将学习率降低10倍
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)


# 测试模型训练之后的预测效果，测试forword
def test_train_pred(coco_model,train_dataloader):
    images,ann = next(iter(train_dataloader))
    targets = []
    for data1 in ann:  # 这个for循环可以舍去
        boxes = []
        target = {}
        labels = []
        for d in data1:
            box=d['bbox']
            box=[box[0],box[1],box[0]+box[2],box[1]+box[3]]
            boxes.append(box)
            labels.append(d['category_id'])
            # convert everything into a torch.Tensor
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.as_tensor(labels, dtype=torch.int64)
        image_id = torch.tensor([data1[0]['image_id']])
        area = (boxes[:, 2]-boxes[:,0]) * (boxes[:, 3]-boxes[:,1])
        print(area)
        # return
        iscrowd = torch.zeros((len(data1),), dtype=torch.int64)
        # suppose all instances are not crowd
        target["boxes"] = boxes
        target["labels"] = labels                   #意思应该是，正样本？？？
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        targets.append(target)

    images = list(image for image in images)
    targets = [{k: v for k, v in t.items()} for t in targets]
    output = coco_model(images,targets)   # Returns losses and detections
    # For inference
    coco_model.eval()
    x = [torch.rand(3, 800, 800), torch.rand(3, 800, 800)]
    predictions = coco_model(x)           # Returns predictions
    print(123)
#test_train_pred(coco_model,train_dataloader)
#模型训练
for epoch in tqdm(range(configs['epoches'])):
    #train for one epoch, printing every 10 iterations
    #train_one_epoch(coco_model, optimizer, train_dataloader, device, epoch, print_freq=10)
    # #update the learning rate
    #lr_scheduler.step()
    #对验证集进行评估
    evaluate(coco_model, val_dataloader, device=device)
#保存模型
state={
    'model':coco_model.state_dict(),
    'optimizer':optimizer.state_dict(),
    'epoch':epoch}
torch.save(state,'minst.pth')