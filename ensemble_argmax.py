"""
Five-fold ensemble

"""


#!/usr/bin/env python
import os
# from networks.swin3d_unetrv2 import SwinUNETR
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai import transforms, data
import nibabel as nib
import scipy.ndimage as ndimage
import argparse
from skimage.measure import label
nib.Nifti1Header.quaternion_threshold = -1e-06

def resize_3d(img, target_size):
    imx, imy, imz = img.shape
    tx, ty, tz = target_size
    zoom_ratio = ( float(tx) / float(imx), float(ty) / float(imy), float(tz) / float(imz))
    img_resampled = ndimage.zoom( img, zoom_ratio, order=0, prefilter=False)
    return img_resampled
def getLargestCC(segmentation):
    labels = label(segmentation)
    assert( labels.max() != 0 ) # assume at least 1 CC
    largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
    return largestCC


#------------------------------------------------------------------------------------------
base_save_pred_dir = '/nfs/masi/zhouy26/22Summer/BodyAtlas/UNesT_inference/runs3-final'
model_names = os.listdir(base_save_pred_dir)

# image testing path
imgpath = '/nfs/masi/zhouy26/22Summer/BodyAtlas/data/QAed/testing/images'

# test_fold_name = 'fold0'
results_folder = '/nfs/masi/zhouy26/22Summer/BodyAtlas/outImages/five_fold-run3final'
if not os.path.exists(results_folder):
    os.makedirs(results_folder)


subject_list = os.listdir(os.path.join(base_save_pred_dir, model_names[0]))

count = 0
for sub in subject_list:
    # sub = '1143_1_target_processed.npy'
    ct_model = len(model_names)
    print(sub)

    infer_outputs = 0.0
    for model_name in model_names:
    # for model_name in ['fold0']:
        # if model_name != test_fold_name:
        #     continue
        print(model_name)
        pred_file = os.path.join(base_save_pred_dir, model_name, sub)
        pred_numpy = np.load(pred_file)
        infer_outputs += pred_numpy

    infer_outputs = infer_outputs / ct_model
    probs = infer_outputs
    # i = i + 1
    # probs = probs.cpu().numpy()
    # print(probs.shape)
    bg_prob = probs[0,0].copy()
    probs[0,0] = 0
    labels = np.argmax(probs, axis=1).astype(np.uint8)[0]  # get discrete lables for each class
    labels[bg_prob>=0.8] = 0





    sub = sub.split('.npy')[0]
    case_name = sub+'.nii'
    output_file = case_name

    # print(f"{sub}: min {np.amin(labels)}   max {np.amax(labels)}")

    original_file = nib.load(os.path.join(imgpath, output_file))
    original_affine = original_file.affine

    target_shape = original_file.shape
    print('target shape: {}'.format(target_shape))
    labels = resize_3d(labels, target_shape)

    # labels = resize_3d(labels, target_shape)

    # --- if needs component analysis -------------------------------------------
    # labels1 = np.zeros((labels.shape[0], labels.shape[1], labels.shape[2]))
    # labels2 = np.zeros((labels.shape[0], labels.shape[1], labels.shape[2]))
    # post_label = np.zeros((labels.shape[0], labels.shape[1], labels.shape[2]))

    # idx1 = np.where(labels == 1)
    # labels1[idx1] = 1

    # # idx2 = np.where(labels == 2)
    # # labels2[idx2] = 1
    # try:
    #     labels1 = getLargestCC(labels1)
    # except AssertionError:
    #     labels1 = labels1

    # try:
    #     labels2 = getLargestCC(labels2)
    # except AssertionError:
    #     labels2 = labels2

    # idx1 = np.where(labels1 == 1)
    # # idx2 = np.where(label2 == 1)

    # post_label[idx1] = 1
    # idx2 = np.where(labels == 2)
    # post_label[idx2] = 2
    # ---------------------------------------------------------------------------------
    try:
        labels = getLargestCC(labels)
    except AssertionError:
        labels = labels    


    count += 1
    print('[{}] label finished {}'.format(count, sub))

    out_name = sub + '_mask.nii'
    nib.save(nib.Nifti1Image(labels.astype(np.uint8), original_affine), os.path.join(results_folder, out_name))
