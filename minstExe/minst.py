#coding=utf-8
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets,transforms
from visdom import Visdom
from torch.utils.data import DataLoader

#先做一个10个图片的标签看以下效果

batch_size=5
learning_rate=0.01
epoches=20

# 5:1:1
train_db=datasets.MNIST('../data',train=True,download=False,transform=transforms.Compose([transforms.ToTensor()]))
train_db,validation_db=torch.utils.data.random_split(train_db,[50000,10000])
test_db=datasets.MNIST('../data',train=False,download=False,transform=transforms.Compose([transforms.ToTensor()]))
#训练和验证是交叉进行的（若是有验证的情况下）
train_loader=DataLoader(train_db,batch_size=batch_size,shuffle=True)
validation_loader=DataLoader(validation_db,batch_size=batch_size,shuffle=True)
test_loader=DataLoader(test_db,batch_size=batch_size,shuffle=False)
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()
    def forward(self,x):
        x=x.view(x.size(0),-1)
        return x
# 定义网络的结构
class Mnist(nn.Module):
    def __init__(self):
        super(Mnist, self).__init__()
        self.net=nn.Sequential(
            Flatten(),
            nn.Linear(784,512),
            nn.ReLU(inplace=True),
            nn.Linear(512,256),
            nn.ReLU(inplace=True),
            nn.Linear(256,128),
            nn.ReLU(inplace=True),
            nn.Linear(128,10),
            nn.ReLU(inplace=True)
        )
    def forward(self,x):
        logits=self.net(x)
        return logits

def main():
    mod=Mnist()
    optimizer=optim.SGD(mod.parameters(),lr=learning_rate)
    #当你想恢复某一阶段的训练（或者进行测试）时，那么就可以读取之前保存的网络模型参数等。
    checkpoint = torch.load('minst.pth')
    mod.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    start_epoch = checkpoint['epoch'] + 1
    loss_fun=nn.CrossEntropyLoss()
    vis=Visdom()
    vis.line([0.],[0.],win='train_loss',opts=dict(title='trai_loss'))
    vis.line([0.],[0.],win='accuracy',opts=dict(title='acc'))
    # vis.line([0.],[0.], win='val_loss', opts=dict(title='val_loss'))
    correct=0
    total_num=0
    global_step=0
    for epoch in range(start_epoch,3):
        for batch_index,(x,y) in enumerate(train_loader):
            # x=x.view(-1,28*28)
            logits=mod(x)
            train_loss=loss_fun(logits,y)
            pred=logits.argmax(dim=1)
            correct+=torch.eq(y,pred).float().sum()
            total_num += x.size(0)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            global_step+=1
            acc=100.*correct/total_num
            #vis.line([train_loss.item()],[global_step],win='train_loss',update='append')
            #vis.line([acc],[global_step],win='accuracy',update='append')
            print('the loss of {:d} step is {:.3f},the accuracy is {:.3f}%'.format(global_step,train_loss.item(),acc))
        #https://www.zhihu.com/question/363144860/answer/951669576(预测时必须使用)
        mod.eval()
        with torch.no_grad():
            val_correct=0
            val_total=0
            for validation_images,validation_label in validation_loader:
                # validation_images=validation_images.view(-1,28*28)
                val_logits=mod(validation_images)
                pred=val_logits.argmax(dim=1)
                val_loss=loss_fun(val_logits,validation_label)
                val_correct+=torch.eq(pred,validation_label).float().sum()
                val_total+=validation_images.size(0)
            # vis.line([val_loss.item()],[global_step],win='val_loss',update='append')
            vis.images(validation_images.view(-1,1,28,28),win='x')
            vis.text(str(pred.detach().cpu().numpy()), win='pred',
                     opts=dict(title='pred'))
            val_acc=100.* val_correct/val_total
            print('the val acc of {:d} epoch is {:.3f}%'.format(epoch,val_acc))
    #保存模型
    # state={
    #     'model':mod.state_dict(),
    #     'optimizer':optimizer.state_dict(),
    #     'epoch':1}
    # torch.save(state,'minst.pth')


if __name__ == '__main__':
    main()
