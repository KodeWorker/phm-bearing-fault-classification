import os
import glob
import numpy as np
import torch
import torch.utils.data as data
from torchvision.datasets import DatasetFolder

labels_map = {
"normal": 0,
"imbalance": 1,
"horizontal-misalignment": 2,
"vertical-misalignment": 3,
"underhang": 4,
"overhang": 5
}

class MAFAULDA(data.Dataset):
    
    def __init__(self, data_dir):
        
        super(MAFAULDA, self).__init__()
        self.data_dir = data_dir
        self.initialize()
    
    def initialize(self):
        
        self.filenames = []
        self.labels = []
        for key, value in labels_map.items():
            filenames = glob.glob(os.path.join(self.data_dir, key, "*.npy"))
            self.filenames += filenames
            self.labels += [value] * len(filenames)
        
        #print(len(self.filenames))
        
    def __getitem__(self, index):
        
        data = np.load(self.filenames[index])
        label = self.labels[index]
        
        data = data.transpose(2,0,1)
        data = torch.from_numpy(np.asarray(data)).float()
        label = torch.from_numpy(np.asarray(label)).float()
        
        return data, label
    
    def __len__(self):
        return len(self.filenames)


def npy_loader(path):
    sample = torch.from_numpy(np.load(path).transpose(2,0,1))
    return sample


if __name__ == "__main__":
    
    data_dir = "../../data/MAFAULDA_XX"
    """
    dataset = MAFAULDA(data_dir)
    
    for x, y in dataset:
        print(x.shape, y.shape)
        break
    """
    
    dataset = DatasetFolder(
    root=data_dir,
    loader=npy_loader,
    extensions='.npy'
    )
    
    for x, y in dataset[-1]:
        print(x.shape, y)
        break