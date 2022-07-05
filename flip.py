import os
import nibabel as nib
import numpy as np
from monai.transforms import Flip

image_dir = '/nfs/masi/zhouy26/22Summer/BodyAtlas/outImages/v6'
subject_list = os.listdir(image_dir)

saving_dir = '/nfs/masi/zhouy26/22Summer/BodyAtlas/flip_images/v6'

if not os.path.exists(saving_dir):
    os.makedirs(saving_dir)

for sub in subject_list:
    
    img = nib.load(os.path.join(image_dir, sub))
    imgnp = np.array(img.dataobj)

    newnp = np.flip(imgnp, axis=0)


    newimgnb = nib.Nifti1Image(newnp,img.affine,img.header)
    nib.save(newimgnb, os.path.join(saving_dir, sub))






