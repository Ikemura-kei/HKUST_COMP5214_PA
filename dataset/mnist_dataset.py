from torch.utils.data import Dataset
import torch
import numpy as np
from utils.read_mnist_data import read_mnist_data

class MnistDataset(Dataset):
    def __init__(self, img_path, label_path):
        super(MnistDataset, self).__init__()
        
        self.imgs, self.labels = read_mnist_data(img_path=img_path, label_path=label_path)
        
    def __getitem__(self, index):
        if torch.is_tensor(index):
            index = index.tolist()
        
        img = torch.from_numpy(self.imgs[index]) / 255.0
        label = torch.from_numpy(np.array(self.labels[index]))
        return img, label
    
    def __len__(self):
        return len(self.imgs)