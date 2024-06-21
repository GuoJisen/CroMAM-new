from pathlib import Path
import random
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset


# load slide patches
class SlideDatasetFromTable(Dataset):
    def __init__(
            self,
            data_file,
            image_dir,
            outcome,
            crop_size,
            transform=None,
    ):

        self.df = data_file
        self.image_dir = image_dir
        self.crop_size = crop_size
        self.transform = transform

        self.outcomes = self.df[[outcome]].to_numpy().astype(float)
        self.ids = self.df['id_patient'].tolist()
        self.files = self.df['file'].tolist()

        try:
            self.random_seeds = self.df['random_id'].tolist()
        except:
            self.random_seeds = np.ones(self.df.shape[0])

    def __len__(self):
        return self.df.shape[0]

    def sample_patch(self, idx):
        idx = idx % self.df.shape[0]
        fname = self.files[idx]
        imgs = Image.open(fname)
        if self.transform is None:
            pass
        else:
            imgs = self.transform(imgs)
        sample = (
            imgs,
            self.ids[idx],
            self.outcomes[idx, :],
        )
        return sample

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.sample_patch(idx)
