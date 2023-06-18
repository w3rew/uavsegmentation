import collections
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import cv2
from pathlib import Path
import albumentations as A
import albumentations.pytorch.transforms as T
import logging
from auxiliary import imread, logger

class LoaderTrainVal:
    def __init__(self, dataset, **kwargs):
        self.train = DataLoader(dataset.train, **kwargs)
        self.val = DataLoader(dataset.val, num_workers=kwargs['num_workers'],
                              shuffle=False)

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
    def __init__(self, dataset, dataset_path, transform=None, mean=None, std=None, shape=None, **kwargs):
        train_dir = dataset_path / dataset.TRAIN_DIR
        val_dir = dataset_path / dataset.VAL_DIR
        if mean is None or std is None:
            logger.warning('Mean and std are not present in config; calculating from the dataset')
            mean, std = dataset.calculate_mean_std(dataset_path)
            logger.info(f'Calculated {mean=}, {std=}')
        self.mean = mean
        self.std = std
        self.train = dataset(train_dir, self.mean, self.std, transform, shape, **kwargs)
        self.val = dataset(val_dir, self.mean, self.std, transform=None, shape=None, **kwargs)


class UAVidCropped(Dataset):
    TRAIN_DIR = 'uavid_train'
    VAL_DIR = 'uavid_val'

    @staticmethod
    def calculate_mean_std(dataset_path, progress=True):
        dirs = [UAVid.TRAIN_DIR, UAVid.VAL_DIR]
        return calculate_mean_std([file for dir_ in dirs for file in (dataset_path / dir_ / 'Images').iterdir()],
                                  progress)

    def _get_images(self, dir_):
        files = []
        seqs = dir_.glob('seq*')

        for seq in seqs:
            files += list((seq / 'Images').iterdir())

        return files

    def _lbl_from_img(self, img_path):
        name = img_path.name

        return img_path.parent.parent / 'TrainId' / name

    def __init__(self, path, mean, std, transform=None, shape=None, **kwargs):
        self.path = Path(path)
        self.files = self._get_images(self.path)
        self.mask_path = self.path / 'TrainId'
        self.mean = [i / 255 for i in mean]
        self.std = [i / 255 for i in std]
        self.shape = shape
        if self.shape is None:
            self.transform = []
        else:
            self.transform = [A.RandomCrop(*self.shape)]
        if transform is not None:
            self.transform.append(transform)
        self.transform += [A.Normalize(self.mean, self.std), T.ToTensorV2()]
        self.transform = A.Compose(self.transform)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file = self.files[idx]
        mask_path = self._lbl_from_img(file)
        img = imread(file)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)

        tmp = self.transform(image=img, mask=mask)
        img, mask = tmp['image'], tmp['mask']


        return f'{file.parent.parent.name}_{file.name}', img, mask[None, ...].long()
