"""
Swin-UNETR
Task09 inference pipeline with separated model

Class 1: Spleen

"""

#!/usr/bin/env python
import os
from monai.networks.nets import UNETR
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai import transforms, data
import nibabel as nib
import scipy.ndimage as ndimage
import argparse
from skimage.measure import label

import numpy as np
import nibabel as nb
import os
from PIL import Image
from collections import defaultdict




brain_color_scheme = '/nfs/masi/zhouy26/22Summer/IHI/brainColorScheme.list'
color_list = []
# left hippocampus idx 17, right hippocampus idx 16, left PHG 98, right PHG 97


# --------------set paths---------
image_dir = '/nfs/masi/zhouy26/22Summer/IHI/data/mni_space_processed'
seg_dir = '/nfs/masi/zhouy26/22Summer/IHI/outImages/five_fold'
overlay_dir = '/nfs/masi/zhouy26/22Summer/IHI/overlay'
#---------------------------------

if not os.path.exists(overlay_dir):
    os.makedirs(overlay_dir)


brain_color_protcol = np.loadtxt(brain_color_scheme,dtype=str,delimiter='\n')


for i in range(len(brain_color_protcol)):
    parts = brain_color_protcol[i].split(' ')
    parts = [part for part in parts if part != '']
    # print(parts)
    # print(i)

    color = []
    color.append(int(parts[2]))
    color.append(int(parts[3]))
    color.append(int(parts[4]))
    color_list.append(color)
print(color_list)    



# count = 0


# for img in os.listdir(image_dir):
#     count += 1
#     # overlay_allorgan_file = os.path.join(overlay_dir, img, 'all_organs', img + '_63.png')
#     # if os.path.isfile(overlay_allorgan_file):
#     #     print('[{}] Exists, skiping {}'.format(count, img))
#     #     continue
#     image_path = os.path.join(image_dir, img)
#     seg_file = os.path.join(seg_dir, img)
#     if os.path.isfile(image_path) and os.path.isfile(seg_file):
#         print('[{}] Processing {}'.format(count, img))
#         imgnb = nb.load(image_path)
#         imgnp = np.array(imgnb.dataobj)



#         segnb = nb.load(seg_file)
#         segnp = np.array(segnb.dataobj)

#         z_range = imgnp.shape[2]
        
#         output_case_dir = os.path.join(overlay_dir, img)
#         if not os.path.isdir(output_case_dir):
#             os.makedirs(output_case_dir)
#         for organ in organ_list:
#             output_case_organ_dir = os.path.join(output_case_dir, organ)
#             if not os.path.isdir(output_case_organ_dir):
#                 os.makedirs(output_case_organ_dir)

#         for organ_idx in range(1,14):
#             current_organ = organ_list[organ_idx-1]
#             #Save all organ multi-labels into all_organs folder
#             if current_organ == 'all_organs':
#                 for i in range(z_range):
#                     #trial 
#                     slice2dnp = imgnp[:,:,i]
                    
#                     slice2dnp = (slice2dnp - slice2dnp.min()) * (255.0 - 0.0) / (slice2dnp.max() - slice2dnp.min())
#                     slice2dnp = slice2dnp.astype(np.uint8)
#                     slice2d = Image.fromarray(slice2dnp).rotate(90)
#                     slice2d = slice2d.convert('RGB')

#                     sliceseg2d = segnp[:,:,i]
#                     sliceseg2d_organs = np.zeros((13,segnp.shape[0], segnp.shape[1]))
               
#                     overlayslice = np.zeros((segnp.shape[0], segnp.shape[1], 3))
                        
#                     overlayslice[:,:,0] = sliceseg2d
#                     overlayslice[:,:,1] = sliceseg2d
#                     overlayslice[:,:,2] = sliceseg2d            
#                     for organ in range(1,13):
#                         indices1 = np.where(overlayslice[:,:,0] == organ)
#                         overlayslice[:,:,0][indices1] = colorpick[organ-1][0]
#                         indices2 = np.where(overlayslice[:,:,1] == organ)
#                         overlayslice[:,:,1][indices2] = colorpick[organ-1][1]
#                         indices3 = np.where(overlayslice[:,:,2] == organ)
#                         overlayslice[:,:,2][indices3] = colorpick[organ-1][2]
                                    
#                     overlayslice_image = Image.fromarray(overlayslice.astype(np.uint8)).rotate(90)                
#                     overlay = Image.blend(slice2d, overlayslice_image, 0.4)
#                     overlay_file = os.path.join(output_case_dir, current_organ, '{}_{}.png'.format(img, i))
#                     overlay.save(overlay_file)
#                     print('[{}] -- {} processed'.format(count, current_organ))