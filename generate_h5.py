import numpy as np
import h5py
import glob
import os
import cv2

import H5_class

data_path = '../data/train/'
number_of_training_images = 200


for folder_name in ['Gaussian','Gaussian_GT']:

    patch_size = 64
    stride = 32
    if folder_name == 'sim':
        patch_size = patch_size * 2
        stride = stride * 2
    
    H5_class.generate_h5f(data_path, folder_name, patch_size, stride, number_of_training_images)
