import os 
from skimage import util
from skimage import io
from imutils import build_montages
import cv2
import numpy as np



# img_list = []
folder_path = "/nfs/masi/zhouy26/22Summer/BodyAtlas/slice/norm98"
# base_path = "/nfs/masi/zhouy26/22Summer/BodyAtlas/figure/intensity_montage"

save_path = "/nfs/masi/zhouy26/22Summer/BodyAtlas/intensity_montage"

if not os.path.exists(save_path):
    os.makedirs(save_path)

# for folder in os.listdir(folder_path): 

#     imgpath = os.path.join(folder_path,folder)
#     img_list = []

#     print("processing {} \n".format(folder))

#     for slice in sorted(os.listdir(imgpath)):

#         # file = io.imread(os.path.join(imgpath,slice))
#         # img_list.append(file)
#         print(slice)
    

# #     if len(img_list) == 30: 
# #         result = util.montage(img_list, grid_shape = (5,6), multichannel=True)
# #     else:
# #         result = util.montage(img_list, grid_shape = (6,8), multichannel=True)
                
# #     save_name = save_path + "/{}.png".format(folder)
            
# #     io.imsave(save_name,result)


# for img in os.listdir(folder_path):

#     #groundtruth label
#     img1_path = os.path.join(folder_path,img)
#     #image

#     img2_path = os.path.join(base_path,"images",img.split('_mask')[0]+".png")
#     #v0
#     img3_path = os.path.join(base_path,"v0",img)
#     #v6
#     img4_path = os.path.join(base_path,"v6",img)

#     img1 = io.imread(img1_path)
#     img2 = io.imread(img2_path)
#     img4 = io.imread(img4_path)

#     img_list = [img2, img1, img4]

#     result = util.montage(img_list, grid_shape = (1,3), multichannel=True)

#     save_name = save_path + "/{}.png".format(img.split(".png")[0])

#     io.imsave(save_name,result)

imglist = []
for img in sorted(os.listdir(folder_path)):
    imgpath = os.path.join(folder_path,img)
    imglist.append(cv2.imread(imgpath))
print(len(imglist))    
    
result = build_montages(imglist, (100,80),(5,6)) 

randomName = str(np.random.randint(100,1000))
for montage in result:
    cv2.imwrite(f"{save_path}/montag_norm98.jpg",montage)



    



