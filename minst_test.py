#coding=utf-8
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from engine import train_one_epoch, evaluate

#https://blog.csdn.net/u013685264/article/details/100564660(faster_rcnn根据自己需要的修改)
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
# replace the classifier with a new one, that has num_classes which is user-defined
num_classes = 24  # 1 class (person) + background
# get number of input features for the classifier
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

# For training
images, boxes = torch.rand(1, 3, 28, 28), torch.rand(1, 1, 4)
labels = torch.randint(1, 23, (4, 1))
images = list(image for image in images)
targets = []
for i in range(len(images)):
    d = {}
    d['boxes'] = boxes[i]
    d['labels'] = labels[i]
    targets.append(d)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)

# the learning rate scheduler decreases the learning rate by 10x every 3 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)

output = model(images, targets)
# For inference
model.eval()
x = [torch.rand(3, 28, 28), torch.rand(3, 28, 28)]
predictions = model(x)
print(predictions[0])
# optionally, if you want to export the model to ONNX:
#torch.onnx.export(model, x, "faster_rcnn.onnx", opset_version = 11)
