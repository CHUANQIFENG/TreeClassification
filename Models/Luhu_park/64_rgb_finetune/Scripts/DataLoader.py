import glob
import random
import os.path
import scipy.io
import numpy as np
from PIL import Image

from Augmentations import *

class DataLoader(object):

    def __init__(self, source_dir, batch_size, target_size_w, target_size_h, stage,augmentations):
        self.batch_size = batch_size
        self.target_size_w = target_size_w
        self.target_size_h = target_size_h
        self.stage = stage
        self.augmentations = augmentations

        self.imgs_RGB = np.zeros((self.batch_size,3,self.target_size_w,self.target_size_h))
        self.labels = np.zeros((self.batch_size,1,1,1))

        if self.stage == 'train': 
            self.dir_RGB = source_dir + '/64_rgb/train'
            self.file_list = glob.glob(self.dir_RGB + '/*.png')
            random.shuffle(self.file_list)
        else:
            self.dir_RGB = source_dir + '/64_rgb/test'
            self.file_list = glob.glob(self.dir_RGB + '/*.png')

        self.cursor = 0

    def next_train_batch(self):
        if self.cursor + self.batch_size > len(self.file_list):
            self.cursor = 0
            random.shuffle(self.file_list)

        index = 0
        
        for sub_cursor in range(self.cursor, self.cursor + self.batch_size):
            # Get file name
            item = self.file_list[sub_cursor]
            item_name = os.path.basename(item)
            #print item_name

            # Get file path
            image_path_RGB = self.dir_RGB + '/' + item_name

            # For RGB file
            array_RGB = np.array(Image.open(image_path_RGB))

            # For Label
            array_label=np.ones((1,1))
            array_str_value=item_name[:item_name.rfind('.')].split('_')[-1:]
            array_label[0,0]= [ int(x) for x in array_str_value ][0]

            # Augmentation
            array_RGB = self.augmentations(array_RGB)

            array_RGB = array_RGB.transpose((2, 0, 1))
            array_label = np.expand_dims(array_label, axis=0)

            self.imgs_RGB[index,...] = array_RGB
            self.labels[index, ...] = array_label

            index=index+1

        self.cursor += self.batch_size
        return self.imgs_RGB, self.labels

    def next_test_batch(self):
        if self.cursor == len(self.file_list):
            self.cursor = 0

        index = 0
        
        for sub_cursor in range(self.cursor, self.cursor + self.batch_size):
            # Get file name
            item = self.file_list[sub_cursor]
            item_name = os.path.basename(item)
            #print item_name

            # Get file path
            image_path_RGB = self.dir_RGB + '/' + item_name

            # For RGB file
            array_RGB = np.array(Image.open(image_path_RGB))

            # For Label
            array_label=np.ones((1,1))
            array_str_value=item_name[:item_name.rfind('.')].split('_')[-1:]
            array_label[0,0]= [ int(x) for x in array_str_value ][0]
            
            array_RGB = array_RGB.transpose((2, 0, 1))
            array_label = np.expand_dims(array_label, axis=0)

            self.imgs_RGB[index,...] = array_RGB
            self.labels[index, ...] = array_label

            index=index+1

        self.cursor += self.batch_size
        return self.imgs_RGB, self.labels