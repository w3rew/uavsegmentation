import collections
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import cv2
from pathlib import Path
import albumentations as A
import albumentations.pytorch.transforms as T

def denormalize(batch, mean, std):
    return batch * std[None, None, ...] + mean[None, None, ...]

class LoaderTrainVal:
    def __init__(self, dataset, **kwargs):
        self.train = DataLoader(dataset.train, **kwargs)
        self.val = DataLoader(dataset.val, batch_size=1, shuffle=False)

def imread(img_path):
    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def calculate_mean_std(imgs, progress=False):
    sums = np.zeros(3, dtype=np.float64)
    sqsums = np.zeros(3, dtype=np.float64)
    c = 0

    if progress:
        iter_ = tqdm(imgs)
    else:
        iter_ = imgs

    for img_path in iter_:
        img = imread(img_path)
        fimg = img.astype(np.float64)
        sums += fimg.sum(axis=(0, 1))
        sqsums += (fimg**2).sum(axis=(0, 1))
        c += img.shape[0] * img.shape[1]

    mean = sums / c
    sqmean = sqsums / c

    return mean, np.sqrt(sqmean - mean**2)



class DatasetTrainVal:
    def __init__(self, dataset, dataset_path, transform=None, mean=None, std=None, **kwargs):
        train_dir = dataset_path / dataset.TRAIN_DIR
        val_dir = dataset_path / dataset.VAL_DIR
        if mean is None or std is None:
            mean, std = dataset.calculate_mean_std(dataset_path)
        self.mean = mean
        self.std = std
        self.train = dataset(train_dir, self.mean, self.std, transform, **kwargs)
        self.val = dataset(val_dir, self.mean, self.std, transform=None, **kwargs)


class UAVid(Dataset):
    TRAIN_DIR = 'uavid_train/seq1'
    VAL_DIR = 'uavid_val/seq16'

    @staticmethod
    def calculate_mean_std(dataset_path, progress=True):
        dirs = [UAVid.TRAIN_DIR, UAVid.VAL_DIR]
        return calculate_mean_std([file for dir_ in dirs for file in (dataset_path / dir_ / 'Images').iterdir()],
                                  progress)

    def __init__(self, path, mean, std, transform=None, shape=None, **kwargs):
        self.path = Path(path)
        self.img_path = self.path / 'Images'
        self.mask_path = self.path / 'TrainId'
        self.files = [file.name for file in self.img_path.iterdir()]
        self.mean = mean
        self.std = std
        self.shape = shape
        self.transform = [A.Normalize(self.mean, self.std), A.Resize(*self.shape)]
        if transform is not None:
            self.transform.append(transform)
        self.transform.append(T.ToTensorV2())
        self.transform = A.Compose(self.transform)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = imread(self.img_path / self.files[idx])
        mask = cv2.imread(str(self.mask_path / self.files[idx]), cv2.IMREAD_UNCHANGED)

        tmp = self.transform(image=img, mask=mask)
        img, mask = tmp['image'], tmp['mask']


        return self.files[idx], img, mask[None, ...].long()
