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

class my_MaxPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False):
        super(my_MaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

    def forward(self, input):
        input = input.transpose(3,1)


        input = F.max_pool2d(input, self.kernel_size, self.stride,
                            self.padding, self.dilation, self.ceil_mode,
                            self.return_indices)
        input = input.transpose(3,1).contiguous()

        return input

    def __repr__(self):
        kh, kw = _pair(self.kernel_size)
        dh, dw = _pair(self.stride)
        padh, padw = _pair(self.padding)
        dilh, dilw = _pair(self.dilation)
        padding_str = ', padding=(' + str(padh) + ', ' + str(padw) + ')' \
            if padh != 0 or padw != 0 else ''
        dilation_str = (', dilation=(' + str(dilh) + ', ' + str(dilw) + ')'
                        if dilh != 0 and dilw != 0 else '')
        ceil_str = ', ceil_mode=' + str(self.ceil_mode)
        return self.__class__.__name__ + '(' \
            + 'kernel_size=(' + str(kh) + ', ' + str(kw) + ')' \
            + ', stride=(' + str(dh) + ', ' + str(dw) + ')' \
            + padding_str + dilation_str + ceil_str + ')'


class my_AvgPool2d(Module):
    def __init__(self, kernel_size, stride=None, padding=0, ceil_mode=False,
                 count_include_pad=True):
        super(my_AvgPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.ceil_mode = ceil_mode
        self.count_include_pad = count_include_pad

    def forward(self, input):
        input = input.transpose(3,1)
        input = F.avg_pool2d(input, self.kernel_size, self.stride,
                            self.padding, self.ceil_mode, self.count_include_pad)
        input = input.transpose(3,1).contiguous()
        return input


    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'kernel_size=' + str(self.kernel_size) \
            + ', stride=' + str(self.stride) \
            + ', padding=' + str(self.padding) \
            + ', ceil_mode=' + str(self.ceil_mode) \
            + ', count_include_pad=' + str(self.count_include_pad) + ')'





class RafDataset(data.Dataset):
    def __init__(self, args, phase, basic_aug=True, transform=None):
        self.raf_path = args.raf_path
        self.phase = phase
        self.basic_aug = basic_aug
        self.transform = transform
        name_c = 0
        label_c = 1
        if phase == 'train':
            # print(self.raf_path)
            dataset = pd.read_csv(os.path.join(self.raf_path, 'train.txt'), sep=' ', header=None)
        else:
            dataset = pd.read_csv(os.path.join(self.raf_path,  'test.txt'), sep=' ', header=None)
            
        self.label = dataset.iloc[:, label_c].values
        images_names = dataset.iloc[:, name_c].values
            

        self.aug_func = [flip_image, add_g]
        self.file_paths = []
        self.clean = (args.label_path == 'list_patition_label.txt')
        # print(self.clean)
        for f in images_names:
            f = f.split(".")[0]
            f += '.jpeg'
            file_name = os.path.join(self.raf_path, phase,f)
            # print(file_name)
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

        return image, label, idx, image1, self.file_paths[idx]
    
    

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    
    expansion = 1
    
    def __init__(self, in_channels, out_channels, stride = 1, downsample = False):
        super().__init__()
                
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size = 3, 
                               stride = stride, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, 
                               stride = 1, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace = True)
        
        if downsample:
            conv = nn.Conv2d(in_channels, out_channels, kernel_size = 1, 
                             stride = stride, bias = False)
            bn = nn.BatchNorm2d(out_channels)
            downsample = nn.Sequential(conv, bn)
        else:
            downsample = None
        
        self.downsample = downsample
        
    def forward(self, x):
        
        i = x
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        
        if self.downsample is not None:
            i = self.downsample(i)
                        
        x += i
        x = self.relu(x)
        
        return x
    

    
class ResNet(nn.Module):
    def __init__(self, block, n_blocks, channels, output_dim):
        super().__init__()
                
        
        self.in_channels = channels[0]
            
        assert len(n_blocks) == len(channels) == 4
        
        self.conv1 = nn.Conv2d(3, self.in_channels, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace = True)
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        
        self.layer1 = self.get_resnet_layer(block, n_blocks[0], channels[0])
        self.layer2 = self.get_resnet_layer(block, n_blocks[1], channels[1], stride = 2)
        self.layer3 = self.get_resnet_layer(block, n_blocks[2], channels[2], stride = 2)
        self.layer4 = self.get_resnet_layer(block, n_blocks[3], channels[3], stride = 2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(self.in_channels, output_dim)
        
    def get_resnet_layer(self, block=BasicBlock, n_blocks=[2,2,2,2], channels=[64, 128, 256, 512], stride = 1):
    
        layers = []
        
        if self.in_channels != block.expansion * channels:
            downsample = True
        else:
            downsample = False
        
        layers.append(block(self.in_channels, channels, stride, downsample))
        
        for i in range(1, n_blocks):
            layers.append(block(block.expansion * channels, channels))

        self.in_channels = block.expansion * channels
            
        return nn.Sequential(*layers)
        
    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        h = x.view(x.shape[0], -1)
        x = self.fc(h)
        
        return x, h
    

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)
    
    
import random
import torch
import numpy as np
from torch.autograd import Variable


##### channel dropping 
def Mask(nb_batch):
    bar = []
    for i in range(7):
        
        foo = [1] * 63 + [0] *  10
        if i == 6:
            foo = [1] * 64 + [0] *  10
        random.shuffle(foo)  #### generate mask
        bar += foo
    bar = [bar for i in range(nb_batch)]
    bar = np.array(bar).astype("float32")
    bar = bar.reshape(nb_batch,512,1,1)
    bar = torch.from_numpy(bar)
    bar = bar.cuda('cuda:{}'.format(args.gpu))
    bar = Variable(bar)
    return bar

###### channel separation and channel diverse loss
def supervisor(x, targets, cnum):
    branch = x
    # print(branch.shape)   [32, 512]
    branch = branch.reshape(branch.size(0),branch.size(1), 1, 1)
    # print(branch.shape) [32 512 1 1]
    branch = my_MaxPool2d(kernel_size=(1,cnum), stride=(1,cnum))(branch)  
    # print(branch.shape) [ 32, 7, 1,1 ]
    branch = branch.reshape(branch.size(0),branch.size(1), branch.size(2) * branch.size(3))
    # print(branch.shape) [32, 7, 1]
    loss_2 = 1.0 - 1.0*torch.mean(torch.sum(branch,2))/cnum # set margin = 3.0
     
    mask = Mask(x.size(0))
    # print(x,mask)
    branch_1 = x.reshape(x.size(0),x.size(1), 1, 1) * mask 
   
    # print(x.size(0), x.reshape(x.size(0),x.size(1), 1, 1).shape, mask.shape)   # 32 [32, 512, 1, 1], [32, 512, 1, 1]
    branch_1 = my_MaxPool2d(kernel_size=(1,cnum), stride=(1,cnum))(branch_1)  
    branch_1 = branch_1.view(branch_1.size(0), -1)
    loss_1 = nn.CrossEntropyLoss()(branch_1, targets)
    return [loss_1, loss_2] 
    
    

class Model(nn.Module):
    def __init__(self, pretrained=True, num_classes=2, drop_rate=0):
        super(Model, self).__init__()
        # # 加载mobileNetV2   
        # # 加载 MobileNetV2 预训练模型
        # self.mobilenet = models.mobilenet_v2(pretrained=pretrained)  

        # # 获取 MobileNetV2 的卷积特征部分
        # self.features = self.mobilenet.features  # 包含大部分的卷积层
        
        # # 创建一个线性层，将1280维的特征映射到512维
        # self.fc1 = nn.Linear(1280, 512)

        # # 最终的分类器层，将512维特征映射到所需的类别数
        # self.classifier = nn.Linear(512, num_classes)

  


        # # 加载Mobile-ViT 
        # 加载 ResNet50 预训练模型
        self.resnet50 = models.resnet50(pretrained=pretrained)
        
        # 提取 ResNet50 的特征部分（去掉最后一个全连接层）
        self.features = nn.Sequential(*list(self.resnet50.children())[:-1])  # 移除最后的分类层
        # 获取最后一层的输入维度（通常是 2048）
        self.fc_in_dim = self.resnet50.fc.in_features
        
        # 定义新的全连接层，将其从 2048 维映射到 512 维
        self.fc = nn.Linear(self.fc_in_dim, 512)
        self.fc_out = nn.Linear(512, num_classes)  # 输出层，用于分类

        # # 激活函数和dropout可以选择性添加
        # self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(p=0.5)  # 可以根据需要调整dropout的比例  

    def forward(self, x, text_features, clip_model, targets, phase='train'):
        # 获取 CLIP 的视觉特征
        with torch.no_grad():            
            image_features = clip_model.encode_image(x)
        
        # x = self.features(x)
            # 遍历 self.features，找到 GAP 层之前的特征
        for i, layer in enumerate(self.features):
            x = layer(x)
            # print('layer:', layer,x.shape)
            # 检查是否到达全局池化层之前
            if isinstance(layer, torch.nn.AdaptiveAvgPool2d):
                break
            feat = x
            # print('feat:', feat.shape)
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

        # Combination
        a = 1
        b = 0.2
        out = a*x_vision+b*x_text

        # Baseline 
        # out = self.fc_out(x)
        
        # print(x_vision[1:7], x_text[1:7])

        if phase=='train':
            return out
        else:
            # return out
            return out, feat   # 热力图
    
    
    

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
parser.add_argument('--raf_path', type=str, default='/home/RAID-5/datasets/Medical_Image/Pneumonia/chest_xray_pneumonia/', help='raf_dataset_path')
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
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 这里为可见的GPU设置设备索引0

# 加载 CLIP 模型
print('device:', device)
clip_model, preprocess = clip.load("ViT-B/32", device=device)


# 标签和情绪的映射关系
label_to_emotion = {
    1: "NORMAL",
    2: "PNEUMONIA"
}

# emotions = [f"{label_to_emotion[i]}" for i in range(1, 8)]
# 使用模板生成情绪描述文本

# emotions = [f"a photo of a {label_to_emotion[i]}" for i in range(1, 8)]  # 89.41 

# emotions = [f"a photo of a {label_to_emotion[i]} Chest x-ray" for i in range(1,3)] # tmux 3-2
emotions = [f"a photo of a {label_to_emotion[i]}" for i in range(1,3)] # 89.26
# emotions = [f"{label_to_emotion[i]}" for i in range(1,3)] # 87.82

def returnCAM_unlabel(feature_conv,a,b):
    import cv2
    # generate the class activation maps upsample to 
    size_upsample = (a,b)  #要保存的特征可视化的图像大小
    width, height = a, b
    bz, nc, h, w = feature_conv.shape
    output_cam = [] 
    cam = feature_conv.sum(1)
    # print(feature_conv.shape, cam.shape)
    cam = cam.reshape(h, w)
    # print(cam.shape)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    cam_img = np.uint8(255 * cam_img)
    output_cam.append(cv2.resize(cam_img, size_upsample))
    output_cam = cv2.applyColorMap(cv2.resize(output_cam[0],(width, height)), cv2.COLORMAP_JET)
    return output_cam

# 热力图保存函数
def save_heatmap_images(images, feature_maps, file_paths):
    _, _, height, width= images.shape   #输入图像宽和高    
    for i in range(len(feature_maps)):                           # **需要改变    feat_3d 与  33行 上面对应
        print(i)
        # features_blods = feat_3d[i,:,:,:].unsqueeze(0).data.cpu()
        feat_3d_input = feature_maps[i,:,:,:].unsqueeze(0).data.cpu().numpy()    # **需要改变    feat_3d 与  33行 上面对应
        heatmap = returnCAM_unlabel(feat_3d_input, height, width)
        img = images[i]                                   # **需要改变    im_data_t 与  32行 上面对应
        # (3,h,w)->(h,w,3)
        img = img.transpose(0,2)
        img = img.transpose(1,0)
        img = np.array(img.cpu())
        # heat + img
        mix = heatmap*1  + 0.2*img*255  #这比例不错     
  
        img_name = file_paths[i]    #  **需要改变    data_t[2]是 文件名， 看你30行target_loader_test遍历后是什么
        img_split = img_name.split('/')
        dir_name = os.path.dirname(file_paths[i])
        print(i,dir_name)  # i  /home/RAID-5/datasets/Medical_Image/Pneumonia/chest_xray_pneumonia/train/PNEUMONIA
        output_dir = dir_name.replace('test', 'Heatmaps_test_0.2text_1vison')
        # if 'NORMAL' in dir_name:  # 判断路径是否包含 'NORMAL'
        #     output_dir = dir_name.replace('NORMAL', 'NORMAL_Heatmaps_train_0.2text_1vison')
        # elif 'PNEUMONIA' in dir_name:  # 判断路径是否 包含 'PNEUMONIA'
        #     output_dir = dir_name.replace('PNEUMONIA', 'PNEUMONIA_Heatmaps_train_0.2text_1vison')
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, img_split[-1])
        cv2.imwrite(output_path, mix) 


# 将情绪描述转换为文本特征
with torch.no_grad():
    text_inputs = clip.tokenize(emotions).to(device)
    text_features = clip_model.encode_text(text_inputs)
    text_features /= text_features.norm(dim=-1, keepdim=True)  # 归一化文本特征
    text_features = text_features.float().to(device)  # 确保在指定设备上


    
def test(model, test_loader, device):
    with torch.no_grad():
        model.eval()

        running_loss = 0.0
        iter_cnt = 0
        correct_sum = 0
        data_num = 0
        data_num = 0


        for batch_i, (imgs1, labels, indexes, imgs2, file_paths) in enumerate(test_loader):
            imgs1 = imgs1.to(device)
            labels = labels.to(device)


            outputs, feature_maps= model(imgs1, text_features, clip_model, labels, phase='test')
            
            # outputs= model(imgs1, text_features, clip_model, labels, phase='test')

            # 热力图
            # save_heatmap_images(imgs1, feature_maps, file_paths)
            # print(outputs, labels)
            loss = nn.CrossEntropyLoss()(outputs, labels)

            iter_cnt += 1
            _, predicts = torch.max(outputs, 1)
            # print(predicts, labels)
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
        # transforms.RandomHorizontalFlip(),
        # begin 加了强数据增强   
        # transforms.RandomVerticalFlip(),  
        # transforms.RandomRotation(degrees=30),  # tmux 3-1
        # transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1), 
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
    model.load_state_dict(torch.load("/home/RAID-5/WH/Open_Code/MERGE/code/best_model/ours_best_Chest_2class_3_0.2text_1vision.pth")['model_state_dict'])
    model.to(device)

    
    # test_acc, test_loss = test(model, test_loader, device)
    test_acc, test_loss = test(model, test_loader, device)

    print('test_acc:', test_acc)




if __name__ == '__main__':
    main()


# 分析这段代码，当我指定gpu, 会占用第0快显卡一部分显存，是哪里的问题？请修改代码，让其不会占用第0号gpu