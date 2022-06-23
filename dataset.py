import cv2
import typing as t
import numpy as np

from pathlib import Path
from einops import repeat
from torch.utils.data import Dataset


class MRIDataset(Dataset):
    def __init__(self, images: t.List[Path], image_shape: t.Tuple[int, int] = (224, 224)):
        self.images = np.array(images)
        self.image_shape = image_shape

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int):
        img_path = self.images[idx]

        img = np.load(img_path)
        img = cv2.resize(img, self.image_shape)

        img = np.asarray(img, dtype=np.float32)
        img /= img.max()
        img = repeat(img, "h w -> c h w", c=1)

        return img

