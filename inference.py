import argparse
import albumentations as A
import logging
import cv2
from auxiliary import CT

logger = logging.getLogger('drone_seg')


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
    def __init__(self, model, resolution, preprocessing_fn):
        self.model = model
        self.resolution = resolution
        self.preprocessing_transform = A.Compose([preprocessing_fn, A.ToTensorV2()])

    def _assemble_from_tiles(self, tiles):
        return np.block(tiles)

    def _inference(self, img):
        preprocessed = self.preprocessing_transform(image=img)

        return preprocessed['image']

    def inference(self, img, color=True):
        if not isinstance(img, np.ndarray):
            raise ValueError(f'img is of type {type(img)}, expected np.ndarray')

        tiles = resize_cut(img, self.resolution)

        mask_tiles = [[self._inference(tile) for tile in row] for row in tiles]
        mask = self._assemble_from_tiles(mask_tiles)

        mask_original_size = cv2.resize(mask, (initial_shape[1], initial_shape[0]))

        if color:
            return CT.transform(mask)
        return mask


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add
