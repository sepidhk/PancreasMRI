import os
from pickletools import uint8
import nibabel as nib
import numpy as np
from PIL import Image



base_dir = "/nfs/masi/zhouy26/22Summer/BodyAtlas/data/QAed/testing/images"
image_dir = "/nfs/masi/zhouy26/22Summer/BodyAtlas/process/testing"
save_dir = "/nfs/masi/zhouy26/22Summer/BodyAtlas/slice/v6"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)


for img in os.listdir(base_dir):

    image_path = os.path.join(base_dir, img)
    imgnb = nib.load(image_path)
    imgnp = np.array(imgnb.dataobj)


    img_name = img.split('.nii')[0]

    z_range = imgnp.shape[2]

    output_casedir = os.path.join(save_dir,img_name)

    if not os.path.exists(output_casedir):
        os.makedirs(output_casedir)

    for i in range(z_range):
        slicenp = (imgnp[:,:,i]*255).astype(np.int8)



        slice2d = Image.fromarray(slicenp).rotate(90)
        slice2d = slice2d.convert('RGB')

        

        file_path = os.path.join(output_casedir, f'{img_name}_{i:02d}.png')

        print(file_path)
        slice2d.save(file_path)





