import os
import glob
import numpy as np
from .slicer import *

      
        
def img_processing(self):
    #files ending in _flair.nii or 
    flairs_path = glob.glob("data/nii_files/*flair.nii")[0]
    masks_path = glob.glob("data/nii_files/*seg.nii*")[0]
    
    vol = load_nii(flairs_path)
    msk = load_nii(masks_path)
    vol = zscore_normalize(vol)
    
    vs, ms = v_to_2d(vol, msk, axis=self.axis)

    return (vs, ms)