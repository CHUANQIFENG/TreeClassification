import math
import numbers
import random
import numpy as np
import scipy.io
from PIL import Image, ImageOps, ImageEnhance

class Compose(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations

    def __call__(self, array_RGB,array_nDSM):

        img_RGB = Image.fromarray(array_RGB,mode='RGB')
        img_nDSM = Image.fromarray(array_nDSM,mode='L')

        for a in self.augmentations:
            img_RGB,img_nDSM = a(img_RGB,img_nDSM)

        return np.array(img_RGB), np.array(img_nDSM)

class RandomHorizontallyFlip(object):
    def __call__(self, img_RGB,img_nDSM):
        if random.random() <= 0.5:
            return img_RGB.transpose(Image.FLIP_LEFT_RIGHT), img_nDSM.transpose(Image.FLIP_LEFT_RIGHT)
        return img_RGB,img_nDSM

class RandomRotate(object):
    def __call__(self, img_RGB,img_nDSM):
        rand= random.random()
        if rand <= 0.25:
            return img_RGB.rotate(90), img_nDSM.rotate(90)
        elif rand <= 0.50:
            return img_RGB.rotate(180), img_nDSM.rotate(180)
        elif rand <= 0.75:
            return img_RGB.rotate(270), img_nDSM.rotate(270)
        else:
            return img_RGB,img_nDSM

class AdjustBrightness(object):
    def __call__(self, img_RGB,img_nDSM):
        rand= random.random()
        enhancer = ImageEnhance.Brightness(img_RGB)
        enhancer_nDSM = ImageEnhance.Brightness(img_nDSM)
        if rand <= 0.33:
            return enhancer.enhance(1.3),enhancer_nDSM.enhance(1.3)
        elif rand <= 0.66:
            return enhancer.enhance(0.7),enhancer_nDSM.enhance(0.7)
        else:
            return img_RGB,img_nDSM

class AdjustContrast(object):
    def __call__(self, img_RGB,img_nDSM):
        rand= random.random()
        enhancer = ImageEnhance.Contrast(img_RGB)
        enhancer_nDSM = ImageEnhance.Contrast(img_nDSM)
        if rand <= 0.33:
            return enhancer.enhance(1.3),enhancer_nDSM.enhance(1.3)
        elif rand <= 0.66:
            return enhancer.enhance(0.7),enhancer_nDSM.enhance(0.7)
        else:
            return img_RGB,img_nDSM
