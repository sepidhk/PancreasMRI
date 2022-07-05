"""
Calculate Dice

"""


#!/usr/bin/env python
import os
from networks.swin3d_unetrv2 import SwinUNETR
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai import transforms, data

import nibabel as nb
import scipy.ndimage as ndimage
import argparse
from skimage.measure import label

import matplotlib.pyplot as plt


def dice(x, y):
    intersect = np.sum(np.sum(np.sum(x * y)))
    y_sum = np.sum(np.sum(np.sum(y)))
    if y_sum == 0:
        return 0.0
    x_sum = np.sum(np.sum(np.sum(x)))
    return 2 * intersect / (x_sum + y_sum)



#=----------

prediction_dir = '/nfs/masi/zhouy26/22Summer/BodyAtlas/flip_images/v6'
groundtruth_dir = '/nfs/masi/zhouy26/22Summer/BodyAtlas/data/QAed/testing/labels'
num_classes = 2
#--------------------

test_sub_dice = []

f = open('/nfs/masi/zhouy26/22Summer/BodyAtlas/dice/v6.txt', 'w')

for image in os.listdir(prediction_dir):
    if image != "img0056_mask.nii":
        imagefile = os.path.join(prediction_dir, image)
        truthfile = os.path.join(groundtruth_dir, image)

        imagenb = nb.load(imagefile)
        truthnb = nb.load(truthfile)

        imagenp = np.array(imagenb.dataobj)
        truthnp = np.array(truthnb.dataobj)

        imagenp = imagenp.astype(np.float32)
        truthnp = truthnp.astype(np.float32)


        # sub_dice = []

        # for i in range(1, num_classes):
        #     score = dice(imagenp==i, truthnp==i)
        #     sub_dice.append(score)

        dice_score = dice(imagenp == 1, truthnp == 1)
        test_sub_dice.append(dice_score)

        # test_sub_dice.append(sub_dice)
        # sub_avg = np.mean(sub_dice)
        # print('Subject: {}, class 1 Dice: {}, class 2 Dice: {}, Avg.: {}'.format(image, sub_dice[0], sub_dice[1], sub_avg))
        # print('Subject: {}, Dice: {}'.format(image, sub_dice[0]))
        print('Subject: {}, Dice: {}'.format(image, dice_score))
        f.write('Subject: {}, Dice: {}'.format(image, dice_score))
        f.write('\n')


# avg_all = np.mean([np.mean(l) for l in test_sub_dice])

plt.boxplot(test_sub_dice)
plt.xlabel('img')
plt.ylim([0,1])
plt.ylabel('Dice')
plt.savefig('/nfs/masi/zhouy26/22Summer/BodyAtlas/figure/box_plotv6.png')

print('Final average all Dice: {}'.format(np.mean(test_sub_dice)))
f.write('Final average all Dice: {}'.format(np.mean(test_sub_dice)))
f.close()
