import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
import pandas as pd
import numpy as np
import os
from PIL import Image

from constants import INPUT_IMAGE_WIDTH, INPUT_IMAGE_HEIGHT

class SegmentationDataset(Dataset):
    """
    This Dataset class loads all the images directly in the init method because the hair remover algorithm is long to process
    so we don't want to apply this transformation 'on the fly'
    """
    def __init__(self, img_dir, annotations_dir, transform_img=None, transform_msk=None):
        self.img_dir = img_dir
        self.annotations = pd.read_csv(annotations_dir)
        all_paths = os.listdir(img_dir)
        self.img_paths = [elt+".jpg" for elt in self.annotations["ID"] if elt+"_seg.png" in all_paths]
        self.msk_paths = [elt.replace('.jpg', '_seg.png') for elt in self.img_paths]
        self.transform_img = transform_img if transform_img else lambda x: x
        self.transform_msk = transform_msk if transform_msk else lambda x: x
        
        N = len(self.img_paths)
        self.images = torch.zeros((N, 3, INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH))
        # The masks has two channels, one for the foreground an the other for the background
        self.masks = torch.zeros((N, 2, INPUT_IMAGE_HEIGHT, INPUT_IMAGE_WIDTH))
        for idx in range(N):
            img = read_image(self.img_dir + self.img_paths[idx])
            fg_msk = read_image(self.img_dir + self.msk_paths[idx])
            bg_msk = 255 - fg_msk
            msk = torch.cat([self.transform_msk(fg_msk), self.transform_msk(bg_msk)], dim=0) # We concat the foreground and background masks
            self.images[idx] = self.transform_img(img)
            self.masks[idx] = msk

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        ann = self.annotations.iloc[idx]["CLASS"] if "CLASS" in self.annotations else None
        if ann:
            return image, mask, ann-1
        return image, mask

class WholeDatasetWithFeatures(Dataset):
    """
    This class loads the dataset and return the image along with extracted features
    """
    def __init__(self, img_dir, annotations_dir, transform=None):
        self.img_dir = img_dir
        self.annotations = pd.read_csv(annotations_dir)
        all_features = list(self.annotations.columns.values)
        self.all_features = [elt for elt in all_features if elt not in ["ID", "CLASS"]]
        self.img_paths = [elt+".jpg" for elt in self.annotations["ID"]]
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = read_image(self.img_dir + img_path)
        ann = self.annotations.iloc[idx]["CLASS"] if "CLASS" in self.annotations else None
        x = torch.tensor(self.annotations.iloc[idx][self.all_features].to_numpy().astype(np.float32))
        if self.transform:
            image = self.transform(image)
        if ann:
            return image, x, ann-1
        return image, x