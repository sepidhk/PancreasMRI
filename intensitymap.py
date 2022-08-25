import nibabel as nib
import os 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


imgdir = '/nfs/masi/zhouy26/22Summer/BodyAtlas/normalize_intensity0-1/normalized_imagestoref'
savedir = '/nfs/masi/zhouy26/22Summer/BodyAtlas/figure/intensity_toref'

if not os.path.exists(savedir):
    os.makedirs(savedir)
count=0


for sub in os.listdir(imgdir):

    img = nib.load(os.path.join(imgdir,sub))
    imgnp = img.get_fdata()

    flat_array = imgnp.flatten()

   
    sns.kdeplot(flat_array)
    plt.xlabel('intensity')
    plt.ylabel('frequency')
    # plt.ylim(0,0.001)
    # plt.xlim(0,120000)
    plt.yscale('log')

   
    savepath = savedir + '/' + f'{count:03d}' + sub.split('nii')[0] + 'png'

    print(f'processing {sub}')
    plt.savefig(savepath)
    count+=1

# plt.savefig('/nfs/masi/zhouy26/22Summer/BodyAtlas/figure/intensity.png')




