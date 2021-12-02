from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset

from utils import CigaretteLighterTransform

class CigaretteLighterDataset(Dataset):
    def __init__(self, mode, x, image_size = 64):
        self.mode = mode
        self.x = x
        self.transform = CigaretteLighterTransform(
            mode = self.mode, 
            size = image_size
        )
    
    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        x = self.x[index]
        image = np.array(Image.open(x))
        transformed = self.transform(image)

        return (transformed['image'].transpose(2, 0, 1) - 127.5) / 127.5
