import os
import nibabel as nib
import numpy as np
from monai.transforms import(
    Compose,
    LoadImaged,
    AddChanneld,
    Spacing,
    Orientation,
    ScaleIntensityRangePercentiles,
    SpatialPad,
    ToTensord,
    CenterSpatialCrop
    )
from monai import data

image_dir = '/nfs/masi/zhouy26/22Summer/BodyAtlas/data/QAed/transform_intensity/training/images'
subject_list = os.listdir(image_dir)
# list = subject_list[:30]

saving_dir = '/nfs/masi/zhouy26/22Summer/BodyAtlas/data/QAed/training/images'

if not os.path.exists(saving_dir):
    os.makedirs(saving_dir)

# transform = Compose(
#     [
#         # Spacing(pixdim=(1.0, 1.0, 2.0), mode=("bilinear")),
#         Orientation(axcodes="RAS"),
#         ScaleIntensityRangePercentiles(
#             lower=0, upper=98,
#             b_min=0.0, b_max=1.0, clip=True,
#         ),
#         SpatialPad(spatial_size=(96, 96, 96)),
#     ]
# )
# transform = ScaleIntensityRangePercentiles(
#             lower=0, upper=98,
#             b_min=0.0, b_max=1.0, clip=True,
#         )
transform = CenterSpatialCrop(roi_size={256, 256, 30})

for sub in subject_list:

    if "scaled_v2" in sub:
        name = sub.split("_scaled_v2")[0] + sub.split("_scaled_v2")[1]

        img = nib.load(os.path.join(image_dir, sub))
        imgnp = np.array(img.dataobj)

        newnp = transform(imgnp)

        newimgnb = nib.Nifti1Image(newnp,img.affine,img.header)
        nib.save(newimgnb, os.path.join(saving_dir, name))


    # img = nib.load(os.path.join(image_dir, sub))
    # imgnp = np.array(img.dataobj)

    # # label = np.where(imgnp != 0.0)
    # # imgnp[label] = 1.0
    # # newnp = imgnp

    # # imgnp = Spacing(pixdim=(1.0, 1.0, 2.0), mode=("bilinear"))(imgnp)
    # # imgnp = Orientation(axcodes="RAS")(imgnp)
    # newnp = transform(imgnp)

    # newimgnb = nib.Nifti1Image(newnp,img.affine,img.header)
    # nib.save(newimgnb, os.path.join(saving_dir, sub))



