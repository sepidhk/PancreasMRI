import os
from pickletools import uint8
import nibabel as nib
import numpy as np
from PIL import Image

import os 
from skimage import util
from skimage import io




base_dir = '/nfs/masi/zhouy26/22Summer/BodyAtlas/normalize_intensity0-1/linear_norm98'
# image_dir = "/nfs/masi/zhouy26/22Summer/BodyAtlas/process/testing"
save_dir = "/nfs/masi/zhouy26/22Summer/BodyAtlas/slice/linear_norm98"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

imglist=[]

for img in sorted(os.listdir(base_dir))[:30]:

    image_path = os.path.join(base_dir, img)
    imgnb = nib.load(image_path)
    imgnp = np.array(imgnb.dataobj)


    img_name = img.split('.nii')[0]

    z_range = imgnp.shape[2]

    output_casedir = os.path.join(save_dir,img_name)

    # if not os.path.exists(output_casedir):
    #     os.makedirs(output_casedir)

    # for i in range(z_range):
    #     slicenp = (imgnp[:,:,i]*255).astype(np.int8)



    #     slice2d = Image.fromarray(slicenp).rotate(90)
    #     slice2d = slice2d.convert('RGB')

    #     file_path = os.path.join(output_casedir, f'{img_name}_{i:02d}.png')

    #     print(file_path)
    #     slice2d.save(file_path)

    slice2dnp = imgnp[:,:,15]
    slice2dnp = (slice2dnp - slice2dnp.min()) * (255.0 - 0.0) / (slice2dnp.max() - slice2dnp.min())
    slice2dnp = slice2dnp.astype(np.uint8)


    slice2d = Image.fromarray(slice2dnp).rotate(90)
    slice2d = slice2d.convert('RGB')
    imglist.append(slice2d)

    file_path = os.path.join(save_dir, f'{img_name}.png')

    print(file_path)
    slice2d.save(file_path)







