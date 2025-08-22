import os
import numpy as np
import tensorflow as tf

from .tf_model import unet_model
from opencv.augmentation import augment

images = 'data/sliced/images'
masks = 'data/sliced/masks'
batch_size = 8
image_size = (256, 256)

