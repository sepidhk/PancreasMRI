import nibabel as nib
import os 
import numpy as np

imgdir = '/nfs/masi/zhouy26/22Summer/BodyAtlas/data/QAed/training/images'
testdir = '/nfs/masi/zhouy26/22Summer/BodyAtlas/data/QAed/training/images'
imglist = [s for s in os.listdir(imgdir)]
testlist = [s for s in os.listdir(testdir)]


# for i in imglist:
#     imagepath = os.path.join(imgdir, i)
#     img = nib.load(imagepath)

#     print("dim: {}".format(img.header['pixdim'][1:4]))

#     imgpy = np.array(img.dataobj)
#     print("intensity max:: {}, min: {}\n".format(np.max(imgpy), np.min(imgpy)))


for i in testlist:
    imagepath = os.path.join(testdir, i)
    img = nib.load(imagepath)

    # print("dim: {}\n".format(img.header['pixdim'][1:4]))

    # imgpy = np.array(img.dataobj)
    # print("intensity max:: {}, min: {}".format(np.max(imgpy), np.min(imgpy)))
    print(f'{i}: shape: {img.shape}' )
