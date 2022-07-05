import nibabel as nib
import os 
import numpy as np

labeldir = '/nfs/masi/zhouy26/22Summer/BodyAtlas/data/QAed/training/labels'
testdir = '/nfs/masi/zhouy26/22Summer/BodyAtlas/data/QAed/testing/labels'
imglist = [s for s in os.listdir(labeldir)]
testlist = [s for s in os.listdir(testdir)]


for i in imglist:
    imagepath = os.path.join(labeldir, i)
    img = nib.load(imagepath)

    imgpy = np.array(img.dataobj)

    idx = np.where(imgpy != 0)
    imgpy[idx] = 1

    new_imagenb = nib.Nifti1Image(imgpy,img.affine,img.header)

    nib.save(new_imagenb,os.path.join(labeldir, i))





for i in testlist:
    imagepath = os.path.join(testdir, i)
    img = nib.load(imagepath)


    imgpy = np.array(img.dataobj)

    idx = np.where(imgpy != 0)
    imgpy[idx] = 1

    new_imagenb = nib.Nifti1Image(imgpy,img.affine,img.header)

    nib.save(new_imagenb,os.path.join(testdir, i))
