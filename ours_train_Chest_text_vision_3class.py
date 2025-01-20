import os
import cv2
import csv
import math
import random
import numpy as np
import pandas as pd
import argparse
import pickle

import torch
import torch.nn as nn
from torchvision import transforms
import torchvision.models as models
import torch.utils.data as data
import torch.nn.functional as F



import torch
import torch.nn as nn
import math


import torch
import cv2
import numpy as np
import random

import pickle
from torch.autograd import Variable

import os
import cv2
import torch.utils.data as data
import pandas as pd
import random
from torchvision import transforms

import sys
sys.path.append('../')

import clip


from torch.nn.modules.module import Module
from torch.nn.modules.utils import _single, _pair, _triple
import torch.nn.functional as F
from torch.nn.parameter import Parameter

class LabelSmoothingCrossEntropy(nn.Module):
    """ NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        super(LabelSmoothingCrossEntropy, self).__init__()
        assert smoothing < 1.0
        self.smoothing = smoothing
        self.confidence = 1. - smoothing

    def forward(self, x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logprobs = F.log_softmax(x, dim=-1)
        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()



class RafDataset(data.Dataset):
    def __init__(self, args, phase, basic_aug=True, transform=None):
        self.raf_path = args.raf_path
        self.phase = phase
        self.basic_aug = basic_aug
        self.transform = transform
        # print(df)
        name_c = 0
        label_c = 1
        if phase == 'train':
            dataset = pd.read_csv(os.path.join(self.raf_path, 'train.txt'), sep=' ', header=None)
        else:
            dataset = pd.read_csv(os.path.join(self.raf_path,  'test.txt'), sep=' ', header=None)
            
        self.label = dataset.iloc[:, label_c].values
        images_names = dataset.iloc[:, name_c].values
        # print(images_names)
        self.aug_func = [flip_image, add_g]
        self.file_paths = []
        self.clean = (args.label_path == 'list_patition_label.txt')
        
        for f in images_names:
            f = f.split(".")[0]
            f += '.jpg'
            file_name = os.path.join(self.raf_path, phase,f)
            self.file_paths.append(file_name)


    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        label = self.label[idx]
        image = cv2.imread(self.file_paths[idx])
        image = image[:, :, ::-1]
        
        
        if not self.clean:    
            image1 = image
            image1 = self.aug_func[0](image)
            image1 = self.transform(image1)

        if self.phase == 'train':
            if self.basic_aug and random.uniform(0, 1) > 0.5:
                image = self.aug_func[1](image)

        if self.transform is not None:
            image = self.transform(image)
        
        if self.clean:
            image1 = transforms.RandomHorizontalFlip(p=1)(image)

        return image, label, idx, image1
    
    



    
    
import random
import torch
import numpy as np
from torch.autograd import Variable



    
    

class Model(nn.Module):
    def __init__(self, pretrained=True, num_classes=3, drop_rate=0):
        super(Model, self).__init__()
        
        self.resnet50 = models.resnet50(pretrained=False)
        
        # 提取 ResNet50 的特征部分（去掉最后一个全连接层）
        self.features = nn.Sequential(*list(self.resnet50.children())[:-1])  # 移除最后的分类层
        
        # 获取最后一层的输入维度（通常是 2048）
        self.fc_in_dim = self.resnet50.fc.in_features
        
        # 定义新的全连接层，将其从 2048 维映射到 512 维
        self.fc = nn.Linear(self.fc_in_dim, 512)
        self.fc_out = nn.Linear(512, num_classes)  # 输出层，用于分类


    def forward(self, x, text_features, clip_model, targets, phase='train'):
        # 获取 CLIP 的视觉特征
        with torch.no_grad():            
            image_features = clip_model.encode_image(x)
        x = self.features(x)

        # 进行全局平均池化，将 (batch_size, 1280, H, W) 转换为 (batch_size, 1280)
        x = x.mean([2, 3])  # [batch_size, 1280]
        
        # 通过第一个全连接层映射到 512 维
        x = self.fc(x)
       
        # 计算图像特征与文本特征的相似度
        x_text = x  @ text_features.T  #  tmux 3-2
        # print(torch.sigmoid(x).shape, image_features.shape)
        
        # 计算图像特征与视觉特征的相似度
        x2 =  x * image_features 
        x_vision = self.fc_out(x2)

        # # Combination
        a = 1
        b = 0.2
        out = a*x_vision+b*x_text

        # Baseline 
        # out = self.fc_osut(x)
        
        # print(x_vision[1:7], x_text[1:7])

        if phase=='train':
            return out
        else:
            return out

    

def add_g(image_array, mean=0.0, var=30):
    std = var ** 0.5
    image_add = image_array + np.random.normal(mean, std, image_array.shape)
    image_add = np.clip(image_add, 0, 255).astype(np.uint8)
    return image_add

def flip_image(image_array):
    return cv2.flip(image_array, 1)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    

    

import torch.nn.functional as F
from torch.autograd import Variable


    

parser = argparse.ArgumentParser()
parser.add_argument('--raf_path', type=str, default='/home/RAID-5/datasets/Medical_Image/Pneumonia/Chest-X-ray_Covid-19_Pneumonia/', help='raf_dataset_path')
parser.add_argument('--resnet50_path', type=str, default='../../resnet50_ft_weight.pkl', help='pretrained_backbone_path')
parser.add_argument('--label_path', type=str, default='list_patition_label.txt', help='label_path')
parser.add_argument('--workers', type=int, default=4, help='number of workers')
parser.add_argument('--batch_size', type=int, default=32, help='batch_size') #默认32
parser.add_argument('--w', type=int, default=7, help='width of the attention map')
parser.add_argument('--h', type=int, default=7, help='height of the attention map')
parser.add_argument('--gpu', type=int, default=0, help='the number of the device')
parser.add_argument('--lam', type=float, default=5, help='kl_lambda')
parser.add_argument('--epochs', type=int, default=60, help='number of epochs')
args = parser.parse_args()


torch.cuda.set_device(args.gpu)
device = torch.device(f'cuda:{args.gpu}')

# 加载 CLIP 模型
print('device:', device)
clip_model, preprocess = clip.load("ViT-B/32", device=device)


# 标签和情绪的映射关系
# label_to_emotion = {
#     1: "NORMAL",
#     2: "PNEUMONIA"
# }
label_to_emotion = {
    1: "COVID19",
    2: "NORMAL",
    3: "PNEUMONIA"

}


emotions = [f"a photo of a {label_to_emotion[i]}" for i in range(1,4)] 


# 将情绪描述转换为文本特征
with torch.no_grad():
    text_inputs = clip.tokenize(emotions).to(device)
    text_features = clip_model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)  # 归一化文本特征
    text_features = text_features.float().to(device)  # 确保在指定设备上



def train(args, model, train_loader, optimizer, scheduler, device):
    running_loss = 0.0
    iter_cnt = 0
    correct_sum = 0
    correct_sum = 0
    
    model.to(device)
    model.train()

    lsce_criterion = LabelSmoothingCrossEntropy(smoothing=0.2)

    total_loss = []
    for batch_i, (imgs1, labels, indexes, imgs2) in enumerate(train_loader):
        imgs1 = imgs1.to(device)
        labels = labels.to(device)
        # print(batch_i)
    
        criterion = nn.CrossEntropyLoss(reduction='none')
        
        output = model(imgs1, text_features, clip_model, labels, phase='train')
        # print(output, labels)
        CE_loss = nn.CrossEntropyLoss()(output, labels)
        
        ## smoooth loss
        lsce_loss = lsce_criterion(output, labels)
        loss1 = 2 * lsce_loss + CE_loss
    

        loss = loss1


        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        iter_cnt += 1
        _, predicts = torch.max(output, 1)
        correct_num = torch.eq(predicts, labels).sum()
        correct_sum += correct_num
        running_loss += loss

    scheduler.step()
    running_loss = running_loss / iter_cnt
    acc = correct_sum.float() / float(train_loader.dataset.__len__())
    return acc, running_loss


    
def test(model, test_loader, device):
    with torch.no_grad():
        model.eval()

        running_loss = 0.0
        iter_cnt = 0
        correct_sum = 0
        data_num = 0
        data_num = 0


        for batch_i, (imgs1, labels, indexes, imgs2) in enumerate(test_loader):
            imgs1 = imgs1.to(device)
            labels = labels.to(device)


            outputs= model(imgs1, text_features, clip_model, labels, phase='test')


            loss = nn.CrossEntropyLoss()(outputs, labels)

            iter_cnt += 1
            _, predicts = torch.max(outputs, 1)

            correct_num = torch.eq(predicts, labels).sum()
            correct_sum += correct_num

            running_loss += loss
            data_num += outputs.size(0)

        running_loss = running_loss / iter_cnt
        test_acc = correct_sum.float() / float(data_num)
        
    return test_acc, running_loss
        
        
        
def main():    
    setup_seed(3407)
    
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.RandomHorizontalFlip(),
        # begin 加了强数据增强   # 性能：tmux (1)3-1  (3)3-2 89.96?     (2)2-1    (2-3)2-2
        # transforms.RandomVerticalFlip(),  # 1
        transforms.RandomRotation(degrees=30),  # 2
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1), # 3
        # transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
        # transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
        # end
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]),
        # transforms.RandomErasing(scale=(0.02, 0.25)) 

        ])
    
    eval_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    
    

    train_dataset = RafDataset(args, phase='train', transform=train_transforms)
    test_dataset = RafDataset(args, phase='test', transform=eval_transforms)
    


    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size,
                                              shuffle=False,
                                              num_workers=args.workers,
                                              pin_memory=False)
    
    
    
    
    model = Model()


    device = torch.device('cuda:{}'.format(args.gpu))
    model.to(device)
#     model = torch.nn.DataParallel(model, device_ids=[0,1,2])

    optimizer = torch.optim.Adam(model.parameters() , lr=0.0002, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    
    best_acc = 0
    for i in range(1, args.epochs + 1):
        train_acc, train_loss = train(args, model, train_loader, optimizer, scheduler, device)
        print('epoch: ', i, 'train_loss:', train_loss, 'train_acc:', train_acc)

        test_acc, test_loss = test(model, test_loader, device)
        if test_acc>best_acc:
            best_acc = test_acc
            torch.save({'model_state_dict': model.state_dict(),}, "./best_model/ours_best_Chest_3class_0.2text_1vision.pth") 
        print('epoch: ', i, 'test_loss:', test_loss, 'test_acc:', test_acc, 'best_acc:',best_acc)
        torch.save({'model_state_dict': model.state_dict(),}, "./best_model/ours_final_Chest_3class_0.2text_1vision.pth") 
        with open('results.txt', 'a') as f:
            f.write(str(i)+'_'+str(test_acc)+'\n')
 


if __name__ == '__main__':
    main()

