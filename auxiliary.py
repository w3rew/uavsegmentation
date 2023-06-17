import matplotlib.pyplot as plt
import torch
import numpy as np
import logging
from UAVidToolKit.colorTransformer import UAVidColorTransformer
import cv2

CT = UAVidColorTransformer()

logger = logging.getLogger('drone_seg')
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.INFO)

def denormalize(batch, mean, std):
    return batch * std[None, :, None, None] + mean[None, :, None, None]

def show_augmentations(loader):
    for name, img, mask in loader:
        photo = denormalize(img, torch.tensor([121.74305257, 126.25519926, 115.68233945]),
                                        torch.tensor([60.41908526, 56.56012129, 60.92941576]))[0, ...].permute(1, 2, 0).long().numpy()
        logger.debug(f'After transform: {img.min()}, {img.max()}')
        logger.debug(f'After transform: {photo.min()}, {photo.max()}')
        mask = mask[0, 0, ...].numpy()
        mask_color = CT.inverse_transform(mask)
        s = np.hstack((photo, mask_color))
        plt.imshow(s)
        plt.imsave('aug.png', s.astype(np.uint8))
        plt.show()

def imread(img_path):
    img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def imwrite(img, img_path):
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(img_path), img)
