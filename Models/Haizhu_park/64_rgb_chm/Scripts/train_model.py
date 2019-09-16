#!/usr/bin/python
# -*- coding:utf-8 -*-

caffe_root = '/caffe-new/'
import sys#
sys.path.insert(0, caffe_root + 'python')

import caffe
import numpy as np

caffe.set_mode_gpu()
caffe.set_device(0)
solver= caffe.get_solver("/TreeClassification/Models/Haizhu_park/64_rgb_chm/resnet50_solver.prototxt")

weights = "/TreeClassification/Models/PretrainedModels/ResNet-50-model.caffemodel"

solver.net.copy_from(weights);
solver.solve()
