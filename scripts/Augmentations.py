# Adapted from https://github.com/ZijunDeng/pytorch-semantic-segmentation/blob/master/utils/joint_transforms.py

import math
import numbers
import random
import numpy as np
from PIL import Image, ImageOps, ImageEnhance

class Compose(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, array_RGB):

        img_RGB = Image.fromarray(array_RGB,mode='RGB')

        for a in self.augmentations:
            img_RGB= a(img_RGB)

        return np.array(img_RGB)

class RandomHorizontallyFlip(object):
    def __call__(self, img_RGB):
        if random.random() <= 0.5:
            return img_RGB.transpose(Image.FLIP_LEFT_RIGHT)
        return img_RGB

class RandomRotate(object):
    def __call__(self, img_RGB):
        rand= random.random()
        if rand <= 0.25:
            return img_RGB.rotate(90)
        elif rand <= 0.50:
            return img_RGB.rotate(180)
        elif rand <= 0.75:
            return img_RGB.rotate(270)
        else:
            return img_RGB

class AdjustBrightness(object):
    def __call__(self, img_RGB):
        rand= random.random()
        enhancer = ImageEnhance.Brightness(img_RGB)
        if rand <= 0.33:
            return enhancer.enhance(1.3)
        elif rand <= 0.66:
            return enhancer.enhance(0.7)
        else:
            return img_RGB

class AdjustContrast(object):
    def __call__(self, img_RGB):
        rand= random.random()
        enhancer = ImageEnhance.Contrast(img_RGB)
        if rand <= 0.33:
            return enhancer.enhance(1.3)
        elif rand <= 0.66:
            return enhancer.enhance(0.7)
        else:
            return img_RGB
