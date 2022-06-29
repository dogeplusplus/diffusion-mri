import typing as t
import numpy as np

from pathlib import Path
from torchvision import transforms
from torch.utils.data import Dataset


class MRIDataset(Dataset):
    def __init__(self, images: t.List[Path], transforms: transforms.Compose = None):
        self.images = np.array(images)
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        img_path = self.images[idx]
        img = np.load(img_path)
        img = img / 255.
        img = np.asarray(img, dtype=np.float32)
        if transforms:
            img = self.transforms(img)

        return img
