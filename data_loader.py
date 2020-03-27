import os,sys
import shutil
import numpy as np
import matplotlib.image as mpimg

def load_image(infilename):
    """Load the image infilename"""
    data = mpimg.imread(infilename)
    return data

def img_crop(im, w, h):
    """Crop patches with width 'w' and height 'h' from the image 'im'"""
    list_patches = []
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    is_2d = len(im.shape) < 3
    for i in range(0,imgheight,h):
        for j in range(0,imgwidth,w):
            if is_2d:
                im_patch = im[j:j+w, i:i+h]
            else:
                im_patch = im[j:j+w, i:i+h, :]
            list_patches.append(im_patch)
    return list_patches

def extract_patch(n, patch_size, imgs):
    """Extract n images (patches_size x patches_size) patches from the list imgs"""
    # Extract patches from input images
    img_patches = [img_crop(imgs[i], patch_size, patch_size) for i in range(n)]
    #gt_patches = [img_crop(gt_imgs[i], patch_size, patch_size) for i in range(n)]

    # Linearize list of patches
    img_patches = np.asarray([img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))])
    #gt_patches =  np.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])
    
    return img_patches #,gt_patches

def load_set(directName, n = np.inf):
    """Load n images from the directory directName"""
    # Loaded a set of images

    files = os.listdir(directName)
    n = min(n, len(files))
    #n = len(files)
    print("Loading " + str(n) + " images")
    imgs = [mpimg.imread(directName + files[i]) for i in range(n)]

    return imgs

def selectPatches(originDir, destinationDir, indices):
    """Select the patches form a given map"""
    for i in indices :
        filePath = originDir + str(i).rstrip() + ".png"

        if not os.path.isfile(filePath):
            print("file does not exist")
        shutil.copy2(filePath,destinationDir)
    print("Done")

def img_crop_window(im, w, h):
    kW=71
    list_patches = []
    
    imgwidth = im.shape[0]
    imgheight = im.shape[1]
    
    pad = (kW - 1) // 2
    im = cv2.copyMakeBorder(im, pad, pad, pad, pad,cv2.BORDER_REFLECT)
    is_2d = len(im.shape) < 3
    
    for i in range(pad,imgheight+pad,h):
        for j in range(pad,imgheight+pad,w):
            if is_2d:
                im_patch = im[j-pad:j+pad+1, i-pad :i+pad +1]
            else:
                im_patch =im[j-pad:j+pad+1, i-pad :i+pad +1,:]
            list_patches.append(im_patch)
    return list_patches