#!/usr/bin/python
# -*- coding:utf-8 -*-

caffe_root = '/home/lucas/caffe-new/'
import sys#
sys.path.insert(0, caffe_root + 'python')

import caffe
import numpy as np

caffe.set_mode_gpu()
caffe.set_device(0)
solver= caffe.get_solver("/home/lucas/TreeClassification/models/Model_64/resnet50_solver.prototxt")

weights = "/home/lucas/TreeClassification/models/PretrainedModels/ResNet-50-model.caffemodel"

solver.net.copy_from(weights);
solver.solve()


