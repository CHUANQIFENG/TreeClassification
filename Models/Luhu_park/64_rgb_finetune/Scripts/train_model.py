#!/usr/bin/python
# -*- coding:utf-8 -*-

caffe_root = '/caffe-new/'
import sys#
sys.path.insert(0, caffe_root + 'python')

import caffe
import numpy as np

caffe.set_mode_gpu()
caffe.set_device(0)
solver= caffe.get_solver("/TreeClassification/Models/Luhu_park/64_rgb_finetune/resnet50_solver.prototxt")

weights = "/TreeClassification/Models/PretrainedModels/Haizhu_park_64_rgb_resnet50_0.00001_iter_160000.caffemodel"

solver.net.copy_from(weights);
solver.solve()


