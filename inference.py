"""
UNETR
Task09 inference pipeline with separated model

Class 1: Spleen

"""

#!/usr/bin/env python
import os
from networks.unest import UNesT
import numpy as np
import torch
from monai.inferers import sliding_window_inference
from monai import data
from networks.unet import UNet

from monai.transforms import (
    Activations,
    AsChannelFirstd,
    AsDiscrete,
    AddChanneld,
    Compose,
    MapTransform,
    CropForegroundd,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandFlipd,
    RandCropByPosNegLabeld,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandSpatialCropd,
    ScaleIntensityRanged,
    Spacingd,
    SpatialPadd,
    RandRotate90d,
    ToTensord,
    RandAffined,
    RandScaleIntensityd,
    RandAdjustContrastd,
    ScaleIntensityRangePercentilesd,
    Resized,
)
import nibabel as nib
import scipy.ndimage as ndimage
import argparse
from skimage.measure import label

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

parser = argparse.ArgumentParser(description='UNETR Training')
parser.add_argument('--imagesTs_path', default=None, type=str)
parser.add_argument('--model_name', default= 'UNesT', type=str)
parser.add_argument('--pred_path', default=None, type=str)
parser.add_argument('--sw_batch_size', default=3, type=int)
parser.add_argument('--overlap', default=0.7, type=float)
parser.add_argument('--fold', default=0, type=int)
parser.add_argument('--rank', default=0, type=int, help='node rank for distributed training')
parser.add_argument('--base_save_pred_dir', default='./UNesT_inference', type=str)
args = parser.parse_args()
torch.backends.cudnn.benchmark = True

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") ## specify the GPU id's, GPU id's start from 0.
### -----------------------------------------------------------------

# # set the output saving folder
# base_save_pred_dir = args.base_save_pred_dir

# set the saved model path
checkpoint_dir = '/nfs/masi/zhouy26/22Summer/BodyAtlas/runs/runs2/fold0_20.v10_unet_size/model.pt'

# set the test image path 
path = '/nfs/masi/zhouy26/22Summer/BodyAtlas/data/QAed/testing/images'

###-------------------------------------------------------------------
results_folder = '/nfs/masi/zhouy26/22Summer/BodyAtlas/UNesT_inference/runs2_v10'
checkpoints = [checkpoint_dir]

# if not os.path.exists(base_save_pred_dir):
#     os.makedirs(base_save_pred_dir)

if not os.path.exists(results_folder):
    os.makedirs(results_folder)
# ----------------------------------------------------------------------------

# set testing list with specific name. 

ids = []
val_files = []
files = os.listdir(path)
for file in files:
    if not file.startswith('.'):
        img_id = file.split('.nii')[0]
        if img_id not in ids:
            ids.append(img_id)
            val_files.append({'label': '',
            'image': [os.path.join(path, img_id+'.nii')]})

# set the testing transformations, probably same with validation transformations
val_transforms = Compose(
    [
        LoadImaged(keys=["image"]),
        AddChanneld(keys=["image"]),
        # Spacingd(keys=["image"], pixdim=(1.5, 1.5, 5.0), mode=("bilinear")),
        # Spacingd(keys=["image"], pixdim=(1.0, 1.0, 2.0), mode=("bilinear")),
        Orientationd(keys=["image"], axcodes="RAS"),
        ScaleIntensityRangePercentilesd(
            keys=["image"], lower=0, upper=95,
            b_min=0.0, b_max=1.0, clip=True,
        ),
        SpatialPadd(keys=["image"], spatial_size=(256, 256, 32)),
        # SpatialPadd(keys=["image"], spatial_size=(96, 96, 96)),

        # ScaleIntensityRangePercentilesd(
        #     keys=["image"], lower=0, upper=98,
        #     b_min=0.0, b_max=1.0, clip=True,
        # ),            
        # CropForegroundd(keys=["image", "label"], source_key="image"),
        ToTensord(keys=["image"]),
    ]

        # print('##############  Inference case {}  ##############'.format(idx))
        # image = batch_data['image'].cuda()
        # affine = batch_data['image_meta_dict']['original_affine'][0].numpy()
        # infer_outputs = 0.0
        # pred = sliding_window_inference(image, img_size, sw_batch_size, model, overlap=overlap_ratio, mode="gaussian")
        # infer_outputs += torch.nn.Softmax(dim=1)(pred) # you may need to use softmax according to each task
        # infer_outputs = infer_outputs.cpu().numpy()
        # case_name = batch_data['image_meta_dict']['filename_or_obj'][0].split('/')[-1].split('.nii')[0]
        # subject_folder = os.path.join(results_folder, case_name)
        # outNUMPY = results_folder + '/' + case_name + '.npy'
        # np.save(outNUMPY, infer_outputs)
)

# import and load models
# img_size = (96,96,96)
img_size = (256,256,32)
# need to set the correct model class

# model = SwinUNETR(in_channels=1,
#                      out_channels=3,
#                      img_size=img_size,
#                      feature_size=48,
#                      patch_size=2,
#                      depths=[2, 2, 2, 2],
#                      num_heads=[3, 6, 12, 24],
#                      window_size=[7, 7, 7])

# model = UNesT(in_channels=1,
#             out_channels=2,
#         ).to(device)

model = UNet( 
    spatial_dims=3,           
    in_channels=1,
    out_channels=2,
    channels=(16,32,64,128,256),
    strides=(2,2,2,2,2),
    ).to(device)
# model = UNETR(
#     in_channels=1,
#     out_channels=14,
#     img_size=(96,96,96),
#     feature_size=32,
#     hidden_size=768,
#     mlp_dim=3072,
#     num_heads=16,
#     pos_embed='perceptron',
#     norm_name='instance',
#     conv_block='store_true',
#     res_block=True,
#     dropout_rate=0.0).to(device)   

ckpt = torch.load(checkpoint_dir, map_location='cpu')
model.load_state_dict(ckpt['state_dict'], strict=True)
#model.load_state_dict(ckpt['state_dict'], strict=False)
model.to(device)
model.cuda()
model.eval()

val_ds = data.Dataset(data=val_files, transform=val_transforms)
val_loader = data.DataLoader(val_ds, batch_size=1, shuffle=False, sampler=None)
overlap_ratio = args.overlap
sw_batch_size = args.sw_batch_size

# run testing iteratively
with torch.no_grad():
    i = 0
    for idx, batch_data in enumerate(val_loader):
        case_name = batch_data['image_meta_dict']['filename_or_obj'][0].split('/')[-1].split('.nii')[0]
        # if idx < 33:
        #     continue

        print('##############  Inference case {}  ##############'.format(idx))
        image = batch_data['image'].cuda()
        affine = batch_data['image_meta_dict']['original_affine'][0].numpy()
        infer_outputs = 0.0
        pred = sliding_window_inference(image, img_size, sw_batch_size, model, overlap=overlap_ratio, mode ="gaussian")
        infer_outputs += torch.nn.Softmax(dim=1)(pred) # you may need to use softmax according to each task
        infer_outputs = infer_outputs.cpu().numpy()
        case_name = batch_data['image_meta_dict']['filename_or_obj'][0].split('/')[-1].split('.nii')[0]
        subject_folder = os.path.join(results_folder, case_name)
        outNUMPY = results_folder + '/' + case_name + '.npy'
        np.save(outNUMPY, infer_outputs)

