import argparse
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import yaml
import architecture
import datasets
import torch.nn as nn
from tqdm import tqdm
import albumentations as A
import logging
import cv2
import numpy as np
from auxiliary import show_augmentations

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
train_transform = A.Compose([A.HorizontalFlip(), A.GridDistortion(p=0.2),
                             A.RandomBrightnessContrast(),
                             A.GaussNoise()])
logger = logging.getLogger('drone_seg')

def to_classes(out):
    classes = np.argmax(out, axis=0)
    return classes

def val_epoch(model, loader, criterion, vis_dir):
    logger.info(f'Starting validation')
    epoch_loss = 0.0
    c = 0
    model.eval()
    with torch.no_grad():
        for name, img, mask in loader.val:
            img = img.to(device)
            mask = mask.to(device)
            out = model(img)
            loss = criterion(out, mask[:, 0, ...])
            epoch_loss += loss.detach().cpu().sum()
            c += img.shape[0]

            classes = to_classes(out.detach().cpu().numpy().squeeze())
            cv2.imwrite(str(vis_dir / name[0]), classes)

        epoch_loss /= c

    return epoch_loss


def train_epoch(model, loader, optim, criterion):
    model.train()
    epoch_loss = 0.0
    c = 0
    for name, img, mask in loader.train:
        optim.zero_grad()
        img = img.to(device)
        mask = mask.to(device)
        out = model(img)
        loss = criterion(out, mask[:, 0, ...])
        loss.backward()
        optim.step()
        epoch_loss += loss.detach().cpu().sum()
        c += img.shape[0]

    epoch_loss /= c

    return epoch_loss


def train(model, cfg, dataset_name, dataset_path, outdir):
    logger.info('Starting model training')
    match dataset_name:
        case 'uavid':
            dataset_cls = datasets.UAVid
        case 'semanticdrone':
            dataset_cls = datasets.SemanticDrone
        case _:
            raise ValueError('Wrong dataset name')

    dataset = datasets.DatasetTrainVal(dataset_cls, dataset_path, transform=train_transform, **cfg['dataset'][dataset_name])
    dataloader = datasets.LoaderTrainVal(dataset, **cfg['dataloader'])

    criterion = nn.CrossEntropyLoss()
    match cfg['train']['optim']['name']:
        case 'adam':
            optim_cls = torch.optim.Adam
        case 'adamw':
            optim_cls = torch.optim.AdamW
        case _:
            raise ValueError('Wrong optimizer name')
    optim = optim_cls(model.parameters(), **cfg['train']['optim']['params'])

    best_val = float('inf')
    best_state_dict = None

    vis_dir = outdir / 'vis'
    vis_dir.mkdir(exist_ok=True)

    for epoch in tqdm(range(cfg['train']['epochs'] + 1)):
        train_loss = train_epoch(model, dataloader, optim, criterion)
        if epoch % 5 == 0:
            val_loss = val_epoch(model, dataloader, criterion, vis_dir)
            logger.info(f'Validation loss {val_loss}')
            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), outdir / 'best.pth')


    return model


def main(args):
    logger.setLevel(logging.DEBUG)
    logger.addHandler(logging.StreamHandler())
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    model = architecture.get_model(cfg['model'])
    if args.input is not None:
        model.load_state_dict(torch.load(args.input, map_location=device))

    model = model.to(device)

    outdir = args.output / args.model_name
    outdir.mkdir(parents=True, exist_ok=True)

    trained = train(model, cfg, args.dataset, args.dataset_path, outdir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Train')
    parser.add_argument('-i', dest='input', type=Path, required=False,
                        help='Path to input model\'s state dict')
    parser.add_argument('-c', dest='config', type=Path, required=True,
                        help='Path to model config')
    parser.add_argument('-o', dest='output', type=Path, required=True,
                        help='Path to output model')
    parser.add_argument('--model_name', required=True,
                        help='Model name')
    parser.add_argument('--dataset', required=True, choices=['uavid', 'semanticdrone'],
                        help='Dataset to train on')
    parser.add_argument('--dataset_path', type=Path, required=True,
                        help='Path to dataset directory')

    args = parser.parse_args()

    main(args)
