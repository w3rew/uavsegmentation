import argparse
import albumentations as A
import albumentations.pytorch.transforms as T
import logging
from auxiliary import CT, imread, imwrite, logger
from pathlib import Path
import architecture
import yaml
from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
import cv2

def cut_tiles(img, resolution):
    if (img.shape[0] % resolution[0] != 0 or
        img.shape[1] % resolution[1] != 0):
        raise ValueError(f'Image shape {img.shape} does not divide'
                         'tile shape {self.resolution}')
    y_tiles = img.shape[0] // resolution[0]
    x_tiles = img.shape[1] // resolution[1]


    tiles = [[None for _ in range(x_tiles)] for _ in range(y_tiles)]

    for i in range(y_tiles):
        for j in range(x_tiles):
            tile = img[resolution[0] * i:resolution[0] * (i + 1),
                       resolution[1] * j: resolution[1] * (j + 1)]
            tiles[i][j] = tile

    return tiles

def resize_cut(img, resolution):
    logger.debug(f'Received image with shape {img.shape}')

    initial_shape = img.shape
    k = img.shape[0] / resolution[0]
    if k >= 1:
        downscale = int(k)
        res = (resolution[0] * downscale, resolution[1] * downscale)
    else:
        res = resolution
    logger.debug(f'Resizing it to {res}')

    resized = cv2.resize(img, (res[1], res[0]))
    logger.debug(f'Resized to {resized.shape}')

    tiles = cut_tiles(resized, resolution)

    return tiles

class ModelInference:
    def __init__(self, model, resolution_k):
        self.model = model
        self.resolution_k = resolution_k
        self.device = next(model.parameters()).device

    def inference(self, img, classes=False):
        self.model.eval()

        initial_shape = img.shape[-2:]

        shape = (img.shape[-2] // self.resolution_k * self.resolution_k,
                 img.shape[-1] // self.resolution_k * self.resolution_k)

        with torch.no_grad():
            resized = F.interpolate(img, shape)
            out = self.model.forward(resized)
            orig_size = F.interpolate(out, initial_shape, mode='nearest-exact')
            if classes:
                orig_size = torch.argmax(orig_size, dim=1, keepdims=True)

        return orig_size


class ModelInferenceImg:
    def __init__(self, model, resolution_k, mean=None, std=None, preprocessing_fn=None):

        self.raw = ModelInference(model, resolution_k)
        if mean and std:
            self.normalize = A.Normalize(mean=[i / 255 for i in mean],
                                     std=[i / 255 for i in std])
        else:
            self.normalize = A.Normalize()

        self.transform = [self.normalize]
        if preprocessing_fn is not None:
            self.transform.append(preprocessing_fn)
        self.transform.append(T.ToTensorV2())
        self.transform = A.Compose(self.transform)

    def inference(self, img, color=True):
        if not isinstance(img, np.ndarray):
            raise ValueError(f'img is of type {type(img)}, expected np.ndarray')

        transformed = self.transform(image=img)['image']
        batched = transformed[None, ...]

        out = self.raw.inference(batched, classes=True)

        out_numpy = out.detach().cpu().numpy()

        mask = out_numpy[0, ...].transpose(1, 2, 0)

        if color:
            mask = CT.inverse_transform(mask[..., 0])

        return mask

def _process(model, img_path, outdir):
    img = imread(img_path)
    out = model.inference(img)


    outpath = outdir / img_path.name
    imwrite(out, outpath)

def main(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    args.outdir.mkdir(exist_ok=True)
    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)
    model = architecture.get_model(cfg['model'])

    model.load_state_dict(torch.load(args.model_weights, map_location=device))

    model = ModelInferenceImg(model, cfg['model']['resolution_k'])

    if args.image.is_file():
        logger.info(f'Processing file {args.image.name}')
        _process(model, args.image, args.outdir)
    elif args.image.is_dir():
        logger.info(f'Processing directory {args.image.name}')
        for file in tqdm(list(args.image.iterdir())):
            _process(model, file, args.outdir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('image', type=Path)
    parser.add_argument('-o', dest='outdir', type=Path)
    parser.add_argument('-m', dest='model_weights', type=Path)
    parser.add_argument('-c', dest='config', type=Path)

    args = parser.parse_args()

    main(args)


