import os
import scipy.io as io
import torch
import torch.utils.data as data
import random

random.seed(1143)

class pair_image_loader(data.Dataset):

    def __init__(self, data_path,tr_path):
        self.data_path = data_path
        self.tr_path = tr_path
        self.data_list = os.listdir(data_path)
        random.shuffle(self.data_list)

        print("训练数据：", len(self.data_list))

    def __getitem__(self, index):
        full_name = self.data_list[index]
        data = io.loadmat(self.data_path + full_name)
        label = io.loadmat(self.tr_path + full_name)

        data = data['data'] / 1.0
        label = label['label'] / 1.0

        data_tensor = torch.tensor(data).float()
        label_tensor = torch.tensor(label).float()

        return {
            'data': data_tensor.unsqueeze(0),
            'label': label_tensor.unsqueeze(0)
        }

    def __len__(self):
        return len(self.data_list)