"""
PyTorch Dataset from CSV (id_code, diagnosis) and CROP image directory.
Image naming: {id_code}.png or {id_code}.jpg
"""
import os
from typing import Tuple, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

from config import IMAGE_SIZE


def _resolve_image_path(img_dir: str, id_code: str) -> Optional[str]:
    for ext in (".png", ".jpg", ".jpeg"):
        p = os.path.join(img_dir, id_code + ext)
        if os.path.isfile(p):
            return p
    return None


class AptosCropDataset(Dataset):
    """CROP images + CSV labels, 5 classes (0-4)."""

    def __init__(
        self,
        csv_path: str,
        img_dir: str,
        transform: Optional[transforms.Compose] = None,
        is_train: bool = True,
    ):
        self.img_dir = img_dir
        self.transform = transform
        self.is_train = is_train

        df = pd.read_csv(csv_path)
        df = df.astype({"diagnosis": "int64"})
        self.samples = []
        for _, row in df.iterrows():
            id_code = row["id_code"]
            label = int(row["diagnosis"])
            path = _resolve_image_path(img_dir, id_code)
            if path is not None:
                self.samples.append((path, label))
        if not self.samples:
            raise FileNotFoundError(
                f"No images found in {img_dir} for CSV {csv_path}. "
                "Check that id_code matches filenames (e.g. id_code.png)."
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def get_transforms(image_size: int, is_train: bool):
    """ImageNet-style normalization; data augmentation when training."""
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    if is_train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
