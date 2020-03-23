import numpy as np
from scipy import signal
from scipy import ndimage

def gauss_kern(kern_size=11, std=3):
    #kern_1d = signal.gaussian(kern_size, std=std).reshape(kern_size, 1)
    kern_1d = signal.windows.gaussian(kern_size, std)
    kern_2d = np.outer(kern_1d, kern_1d)
    return kern_2d

def convolve(img, kernel):
    out = np.copy(img) / np.max(img)
    out[:,:,0] = signal.convolve(img[:,:,0], kernel, mode="same")
    out[:,:,1] = signal.convolve(img[:,:,1], kernel, mode="same")
    out[:,:,2] = signal.convolve(img[:,:,2], kernel, mode="same")
    return (255 * out / np.max(out)).astype(uint8)