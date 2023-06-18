import argparse
from pathlib import Path
import torch
import yaml
import architecture
import datasets
import torch.nn as nn
from tqdm import tqdm
import albumentations as A
import logging
import cv2
import numpy as np
import torchmetrics.classification as tc
import segmentation_models_pytorch as smp
from auxiliary import show_augmentations, logger, CT, imwrite
from inference import ModelInference

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
train_transform = A.Compose([A.HorizontalFlip(), A.GridDistortion(p=0.2),
                             A.RandomBrightnessContrast(),
                             A.GaussNoise()])

def val_epoch(inference_model, loader, criterion, *, vis_dir=None, progress=False):
    logger.info(f'Starting validation')
    epoch_loss = 0.0
    c = 0
    if progress:
        iter_ = tqdm(loader.val)
    else:
        iter_ = loader.val
    with torch.no_grad():
        for name, img, mask in iter_:
            img = img.to(device)
            mask = mask.to(device)
            out = inference_model.inference(img, classes=False)
            loss = criterion(out, mask[:, 0, ...])
            epoch_loss += loss.detach().cpu().sum()
            c += img.shape[0]

            if vis_dir:
                classes = out.detach().cpu().numpy()[0, 0, ...]
                color = CT.inverse_transform(classes)
                imwrite(color, vis_dir / name[0])

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

def wheels(cfg, dataset_name, dataset_path):
    model = architecture.get_model(cfg['model'])
    if args.input is not None:
        logger.info('Loading model')
        model.load_state_dict(torch.load(args.input, map_location=device))

    model = model.to(device)

    inference_model = ModelInference(model, cfg['model']['resolution_k'])


    match dataset_name:
        case 'uavid':
            dataset_cls = datasets.UAVidCropped
            logger.info('UAVid dataset')
        case 'semanticdrone':
            dataset_cls = datasets.SemanticDrone
        case _:
            raise ValueError('Wrong dataset name')

    dataset = datasets.DatasetTrainVal(dataset_cls, dataset_path, transform=train_transform,
                                       shape=cfg['model']['shape'],
                                       **cfg['dataset'][dataset_name])
    dataloader = datasets.LoaderTrainVal(dataset, **cfg['dataloader'])

    logger.info(f'Train loss: {cfg["train"]["loss"]}')
    match cfg['train']['loss']:
        case 'cross_entropy':
            train_criterion = nn.CrossEntropyLoss()
        case 'jaccard':
            train_criterion = smp.losses.JaccardLoss('multiclass')
        case 'dice':
            train_criterion = smp.losses.DiceLoss('multiclass')
        case 'lovasz':
            train_criterion = smp.losses.LovaszLoss('multiclass')
        case 'focal':
            train_criterion = smp.losses.FocalLoss('multiclass')

    val_index = tc.MulticlassJaccardIndex(cfg['model']['params']['classes']).to(device)
    val_criterion = lambda a, b: 1 - val_index(a, b)
    logger.info(f'Val loss: Exact Jaccard loss')


    match cfg['train']['optim']['name']:
        case 'adam':
            optim_cls = torch.optim.Adam
        case 'adamw':
            optim_cls = torch.optim.AdamW
        case _:
            raise ValueError('Wrong optimizer name')
    optim = optim_cls(model.parameters(), **cfg['train']['optim']['params'])

    return model, inference_model, dataloader, train_criterion, val_criterion, optim



def train(model, inference_model, dataloader, train_criterion, val_criterion, optim, cfg, outdir):
    logger.info('Starting model training')

    best_val = float('+inf')
    best_state_dict = None

    vis_dir = outdir / 'vis'
    vis_dir.mkdir(exist_ok=True)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', patience=4)

    for epoch in tqdm(range(cfg['epochs'] + 1)):
        train_loss = train_epoch(model, dataloader, optim, train_criterion)
        if epoch % 5 == 0:
            val_loss = val_epoch(inference_model, dataloader, val_criterion, vis_dir=vis_dir)
            scheduler.step(val_loss)
            logger.info(f'Validation index {1 - val_loss}')
            if val_loss < best_val:
                best_val = val_loss
                torch.save(model.state_dict(), outdir / 'best.pth')


    return model


def main(args):
    if args.debug:
        logger.setLevel(logging.DEBUG)
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    model, inference_model, dataloader, train_criterion, val_criterion, optim = wheels(cfg, args.dataset,
                                                                      args.dataset_path)

    if args.validate_only:
        logger.info('Validating the model')
        val_loss = val_epoch(inference_model, dataloader, val_criterion, progress=True)
        print(f'Validation loss is {val_loss}')
        return

    outdir = args.output / args.model_name
    outdir.mkdir(parents=True, exist_ok=True)

    trained = train(model, inference_model, dataloader, train_criterion,
                    val_criterion, optim, cfg['train'], outdir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Train')
    parser.add_argument('-i', dest='input', type=Path, required=False,
                        help='Path to input model\'s state dict')
    parser.add_argument('-c', dest='config', type=Path, required=True,
                        help='Path to model config')
    parser.add_argument('-o', dest='output', type=Path,
                        help='Path to output model')
    parser.add_argument('--model_name',
                        help='Model name')
    parser.add_argument('--dataset', choices=['uavid', 'semanticdrone'],
                        help='Dataset to train on')
    parser.add_argument('--dataset_path', type=Path, required=True,
                        help='Path to dataset directory')
    parser.add_argument('-d', dest='debug', action='store_true',
                        help='Debug flag')
    parser.add_argument('--validate_only', action='store_true',
                        help='Only validate the model')

    args = parser.parse_args()

    main(args)
