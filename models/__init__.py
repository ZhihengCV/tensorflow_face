# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from models import inception_model
from models import alexnet_model
from models import vgg_model
from models import densenet_model

model_map = {
    'inception_v3': inception_model,
    'alexnet': alexnet_model,
    'vgg': vgg_model,
    'densenet': densenet_model
}