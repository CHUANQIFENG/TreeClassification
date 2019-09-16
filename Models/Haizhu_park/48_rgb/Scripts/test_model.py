#!/usr/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import glob
import matplotlib.pyplot as plt
import os.path
import json
import scipy
import argparse
import scipy.io
import math
import pylab
caffe_root = '/caffe-new/'
import sys
sys.path.insert(0, caffe_root + 'python')

import caffe

# Import arguments
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, required=True)
parser.add_argument('--weights', type=str, required=True)
parser.add_argument('--iter', type=int, required=True)
args = parser.parse_args()

caffe.set_mode_gpu()
caffe.set_device(0)

net = caffe.Net(args.model,
                args.weights,
                caffe.TEST)


dir_RGB = '/TreeClassification/Data/Haizhu_park/48_rgb/test'
file_list = glob.glob(dir_RGB + '/*.png')

for i in range(0, args.iter):

    net.forward()

    item = file_list[i]
    item_name = os.path.basename(item)

    data_Label = net.blobs['data_Label'].data
    data_Label = np.squeeze(data_Label[0,0,0,0])

    predicted = net.blobs['prob'].data
    predicted = np.squeeze(predicted[0])
    predicted = np.argmax(predicted,axis=0)

    print item_name,data_Label,predicted

print 'Success!'