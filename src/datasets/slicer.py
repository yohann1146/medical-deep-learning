import os
import numpy as np
import nibabel
from skimage.transform import resize

def zscore_normalize(v, e=1e-6):
    mu, sigma = np.mean(v), np.std(v)
    return (mu-sigma) / (sigma+e)   #normal z distr

def v_to_2d(v, mask=None, axis=2):
    if axis==0:
        vs=[v[i, :, :] for i in range(v.shape[0])]
        ms=[mask[i, :, :] if mask is not None else None for i in range(v.shape[0])]
        
    if axis==1:
        vs=[v[:, i, :] for i in range(v.shape[1])]
        ms=[mask[:, i, :] if mask is not None else None for i in range(v.shape[1])]
        
    else:
        vs=[v[:, :, i] for i in range(v.shape[2])]
        ms=[mask[:, :, i] if mask is not None else None for i in range(v.shape[2])]
        
    return vs, ms   # 3d volume and mask to 2d slices
    
def load_nii(path):
    img = nibabel.load(path)
    arr = img.get_fdata().astype(np.float32)
    return arr    #nifti format is kinda weird

def fit_resize(img, target=(256, 256)):
    return resize(img, target, order=1, mode='reflect', anti_aliasing=True, preserve_range=True).astype(np.float32)

