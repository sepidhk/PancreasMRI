from genericpath import exists
import numpy as np
import nibabel as nb
import os
from PIL import Image


colorpick = [[255,30,30],[255,245,71],[112,255,99],[9,150,37],[30,178,252],[132,0,188],\
    [255,81,255],[158,191,9],[255,154,2],[102,255,165],[0,242,209],[255,0,80],[40,0,242]]
#Spleen: red,right kid: yellow, left kid green, gall:sky blue, eso:blue,liver:lg blue
#sto:pink,aorta: purple,IVC, potal vein: orange, pancreas: favor, adrenal gland
organ_list = ['spleen', 'right_kigney', 'left_kidney', 'gallbladder', 'esophagus', 'liver',\
                'stomach', 'aorta', 'IVC', 'veins', 'pancreas', 'adrenal_gland', 'all_organs']


# --------------set paths---------
image_dir = '/nfs/masi/zhouy26/22Summer/BodyAtlas/data/QAed/testing/images'
seg_dir = '/nfs/masi/zhouy26/Transformer_based/UNETR/outUNETRImages'
overlay_dir = './overlay_img'
#---------------------------------

if not os.path.exists(overlay_dir):
    os.makedirs(overlay_dir)


count = 0
for img in os.listdir(image_dir):
    count += 1

    image_path = os.path.join(image_dir, img)

    seg_file = os.path.join(seg_dir, img)

    if os.path.isfile(image_path) and os.path.isfile(seg_file):
        print('[{}] Processing {}'.format(count, img))
        imgnb = nb.load(image_path)
        imgnp = np.array(imgnb.dataobj)

        # soft tissue windowing
        idx = np.where(imgnp < -175)
        imgnp[idx[0], idx[1], idx[2]] = -175 # set minmum to -175
        idx = np.where(imgnp > 275)
        imgnp[idx[0], idx[1], idx[2]] = 275 # set maximum to 275
        imgnp = (imgnp - imgnp.min()) * (1.0 - 0.0) / (imgnp.max() - imgnp.min())

        segnb = nb.load(seg_file)
        segnp = np.array(segnb.dataobj)

        z_range = imgnp.shape[2]
        


        output_case_dir = os.path.join(overlay_dir, img)
        if not os.path.isdir(output_case_dir):
            os.makedirs(output_case_dir)

        output_original_dir = os.path.join(output_case_dir, 'original')
        if not os.path.isdir(output_original_dir):
            os.makedirs(output_original_dir)

        for organ in organ_list:
            output_case_organ_dir = os.path.join(output_case_dir, organ)
            if not os.path.isdir(output_case_organ_dir):
                os.makedirs(output_case_organ_dir)

        for organ_idx in range(1,14):
            current_organ = organ_list[organ_idx-1]
            #Save all organ multi-labels into all_organs folder
            if current_organ == 'all_organs':
                for i in range(z_range):
                    slice2dnp = (imgnp[:,:,i] - imgnp[:,:,i].min()) * (255.0 - 0.0) / (imgnp[:,:,i].max() - imgnp[:,:,i].min())

                    slice2d = Image.fromarray(slice2dnp.astype(np.uint8)).rotate(90)
                    slice2d = slice2d.convert('RGB')

                    sliceseg2d = segnp[:,:,i]
                    #13 channels
                    sliceseg2d_organs = np.zeros((12,segnp.shape[0], segnp.shape[1]))
               
                    overlayslice = np.zeros((segnp.shape[0], segnp.shape[1], 3))
                        
                    overlayslice[:,:,0] = sliceseg2d
                    overlayslice[:,:,1] = sliceseg2d
                    overlayslice[:,:,2] = sliceseg2d


                    for organ in range(1,13):
                        indices1 = np.where(overlayslice[:,:,0] == organ)
                        overlayslice[:,:,0][indices1] = colorpick[organ-1][0]
                        indices2 = np.where(overlayslice[:,:,1] == organ)
                        overlayslice[:,:,1][indices2] = colorpick[organ-1][1]
                        indices3 = np.where(overlayslice[:,:,2] == organ)
                        overlayslice[:,:,2][indices3] = colorpick[organ-1][2]
                                    
                    overlayslice_image = Image.fromarray(overlayslice.astype(np.uint8)).rotate(90)                
                    overlay = Image.blend(slice2d, overlayslice_image, 0.4)
                    overlay_file = os.path.join(output_case_dir, current_organ, '{}_{}.png'.format(img, i))
                    overlay.save(overlay_file)
                    print('[{}] -- {} processed'.format(count, current_organ))

                    original_file = os.path.join(output_case_dir, 'original', '{}_{}.png'.format(img, i))
                    slice2d.save(original_file)






























