'''
A data augmentation pipeline for datasets in bird's eye view, i.e. where there is
no canonical "up" or "down" in the images.

Copyright (C) 2018 Pierluigi Ferrari

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
'''

from __future__ import division
import numpy as np

from data_generator.classification_photometric_ops import ConvertColor, ConvertDataType, ConvertTo3Channels, RandomBrightness, RandomContrast, RandomHue, RandomSaturation, Standardize
from data_generator.classification_geometric_ops import Resize, RandomFlip, RandomRotate

class DataAugmentationSatellite:
    '''
    A data augmentation pipeline for datasets in bird's eye view, i.e. where there is
    no "up" or "down" in the images.

    Applies a chain of photometric and geometric image transformations. For documentation, please refer
    to the documentation of the individual transformations involved.
    '''

    def __init__(self,
                 resize=False,
                 subtrahend=127.5,
                 divisor=127.5,
                 random_brightness=(-48, 48, 0.5),
                 random_contrast=(0.5, 1.8, 0.5),
                 random_saturation=(0.5, 1.8, 0.5),
                 random_hue=(18, 0.5),
                 random_flip=0.5,
                 random_rotate=([90, 180, 270], 0.5),
                 background=(0,0,0)):

        self.background = background

        # Utility transformations
        self.convert_to_3_channels  = ConvertTo3Channels() # Make sure all images end up having 3 channels.
        self.convert_RGB_to_HSV     = ConvertColor(current='RGB', to='HSV')
        self.convert_HSV_to_RGB     = ConvertColor(current='HSV', to='RGB')
        self.convert_to_float32     = ConvertDataType(to='float32')
        self.convert_to_uint8       = ConvertDataType(to='uint8')
        if resize:
            self.resize             = Resize(height=resize[0],
                                             width=resize[1])
        self.standardize            = Standardize(subtrahend=subtrahend, divisor=divisor)

        # Photometric transformations
        self.random_brightness      = RandomBrightness(lower=random_brightness[0], upper=random_brightness[1], prob=random_brightness[2])
        self.random_contrast        = RandomContrast(lower=random_contrast[0], upper=random_contrast[1], prob=random_contrast[2])
        self.random_saturation      = RandomSaturation(lower=random_saturation[0], upper=random_saturation[1], prob=random_saturation[2])
        self.random_hue             = RandomHue(max_delta=random_hue[0], prob=random_hue[1])

        # Geometric transformations
        self.random_horizontal_flip = RandomFlip(dim='horizontal', prob=random_flip)
        self.random_rotate          = RandomRotate(angles=random_rotate[0], prob=random_rotate[1])

        # Define the processing chain.
        self.transformations = [self.convert_to_3_channels,
                                self.convert_to_float32,
                                self.random_brightness,
                                self.random_contrast,
                                self.convert_to_uint8,
                                self.convert_RGB_to_HSV,
                                self.convert_to_float32,
                                self.random_saturation,
                                self.random_hue,
                                self.convert_to_uint8,
                                self.convert_HSV_to_RGB,
                                self.random_horizontal_flip,
                                self.random_rotate,
                                self.standardize]

    def __call__(self, image):

        for transform in self.transformations:
            image = transform(image)
        return image
