import glob
import pandas as pd
import numpy as np
import pickle
from PIL import Image
from torch.utils.data.dataloader import DataLoader
from tqdm.notebook import tqdm

import cv2

from sklearn.preprocessing import normalize as sknormalize
from sklearn.decomposition import PCA

import torch

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

import torchvision.models as models
import torchvision.transforms as transforms
from torch.autograd import Variable
from torch.utils.data.dataset import Dataset
import os


class QRDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.img_path = os.listdir(folder_path)
        self.img_path.sort(key=lambda x: x.lower())
        self.folder_path = folder_path
        self.img_label = np.zeros(len(self.img_path))

        if transform is not None:
            self.transform = transform
        else:
            self.transform = None

    def __getitem__(self, index):
        img = Image.open(self.folder_path + '/' + self.img_path[index])

        if self.transform is not None:
            img = self.transform(img)

        return img, self.img_path[index]

    def __len__(self):
        return len(self.img_path)


class Img2Vec():

    def __init__(self, model='resnet-18', layer='default', layer_output_size=512, batch_size=10):
        """ Img2Vec
        :param model: String name of requesdepictingted model
        :param layer: String or Int depending on model.
        :param layer_output_size: Int  the output size of the requested layer
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.layer_output_size = layer_output_size
        self.model_name = model
        self.batch_size = batch_size
        self.model, self.extraction_layer = self._get_model_and_layer(model, layer)

        self.model = self.model.to(self.device)

        self.model.eval()
        # 训练完train_datasets之后，model要来测试样本了。在model(test_datasets)之前，需要加上model.eval().
        # 否则的话，有输入数据，即使不训练，它也会改变权值

        self.transformer = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # normalized方法用于对数据的均值与标准差进行标准化，三维对应图像的三个通道
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def get_vec(self, path):
        """ Get vector embedding from PIL image
        :param path: Path of image dataset
        :returns: Numpy ndarray
        """
        # if not isinstance(path, list):
        #     path = [path]

        # batch_size 即为一次训练的数据个数，shuffle 每训练一次打乱数据集，否则就是按顺序训练
        # num_workers 用于加载数据的子进程数量
        data_loader = torch.utils.data.DataLoader(QRDataset(path, self.transformer), batch_size=self.batch_size,
                                                  shuffle=False, num_workers=16)

        my_embedding = []

        # hook function
        # hook函数用于获取中间层输出（特征），详见https://zhuanlan.zhihu.com/p/75054200
        def append_data(module, input, output):
            my_embedding.append(output.clone().detach().cpu().numpy())

        # no_grad()就是不再计算梯度、反向传播，也就是不再对模型进行训练，只是不断前向传播进行预测
        # 这种模式可以帮助节省内存空间
        with torch.no_grad():
            for batch_data in tqdm(data_loader):
                batch_x, batch_y = batch_data
                print(batch_y)
                if torch.cuda.is_available():
                    batch_x = Variable(batch_x, requires_grad=False).cuda()
                else:
                    batch_x = Variable(batch_x, requires_grad=False)
                # 注册hook函数，在数据经过当前层之后便获取输出（特征），添加到当前的嵌入列表my_embedding
                h = self.extraction_layer.register_forward_hook(append_data)
                h_x = self.model(batch_x)
                h.remove()
                del h_x

        # np.vstack:按垂直方向（行顺序）堆叠数组构成一个新的数组 如[[1,4],[2,5],[3,6]] -> [[1,2,3],[4,5,6]]
        # 这里是因为因为网络的输出是一个列向量，将其旋转成行向量
        my_embedding = np.vstack(my_embedding)
        if self.model_name == 'alexnet':
            return my_embedding[:, :]
        else:
            return my_embedding[:, :, 0, 0]

    def _get_model_and_layer(self, model_name, layer):
        """ Internal method for getting layer from model
        :param model_name: model name such as 'resnet-18'
        :param layer: layer as a string for resnet-18 or int for alexnet
        :returns: pytorch model, selected layer
        """
        if model_name == 'resnet-18':
            model = models.resnet18(pretrained=True)  # pretrained=True 会加载预先训练好的参数
            # models 为 import torchvision.models as models
            if layer == 'default':
                layer = model._modules.get('avgpool')
                self.layer_output_size = 512
            else:
                # _modules 用来保存注册的 Module 对象，_modules是一个字典（哈希表）对象
                layer = model._modules.get(layer)

            return model, layer

        elif model_name == 'resnet-50':
            model = models.resnet50(pretrained=True)
            if layer == 'default':
                layer = model._modules.get('avgpool')
                self.layer_output_size = 2048
            else:
                layer = model._modules.get(layer)

            return model, layer

        elif model_name == 'alexnet':
            model = models.alexnet(pretrained=True)
            if layer == 'default':
                layer = model.classifier[-2]
                self.layer_output_size = 4096
            else:
                layer = model.classifier[-layer]

            return model, layer

        else:
            raise KeyError('Model %s was not found' % model_name)


if __name__ == '__main__':
    net = Img2Vec()  # Img2Vec 是封装好网络的类
    feature = net.get_vec("video")  # video是关键帧所在的文件夹
    np.save("feature_from_nn_image", feature)
    # transformer = transforms.Compose([
    #     transforms.Resize((224, 224)),
    #     transforms.ToTensor(),
    #     # normalized方法用于对数据的均值与标准差进行标准化，三维对应图像的三个通道
    #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    # ])
    # data = QRDataset("video")
    # dataloader = DataLoader(data, batch_size=10, shuffle=True)
