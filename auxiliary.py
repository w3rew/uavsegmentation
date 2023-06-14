import matplotlib.pyplot as plt
import torch
import numpy as np
import logging
from UAVidToolKit.colorTransformer import UAVidColorTransformer
ct = UAVidColorTransformer()

logger = logging.getLogger('drone_seg')

def denormalize(batch, mean, std):
    return batch * std[None, :, None, None] + mean[None, :, None, None]

def show_augmentations(loader):
    for name, img, mask in loader:
        photo = denormalize(img, torch.tensor([121.74305257, 126.25519926, 115.68233945]),
                                        torch.tensor([60.41908526, 56.56012129, 60.92941576]))[0, ...].permute(1, 2, 0).long().numpy()
        logger.debug(f'After transform: {img.min()}, {img.max()}')
        logger.debug(f'After transform: {photo.min()}, {photo.max()}')
        mask = mask[0, 0, ...].numpy()
        mask_color = ct.inverse_transform(mask)
        plt.imshow(np.hstack((photo, mask_color)))
        plt.show()

