"""
    change name of the file
"""

import os 
import csv


#imgdir = '/nfs/masi/zhouy26/22Summer/BodyAtlas/data/QAed/images'
imgdir = '/nfs/masi/zhouy26/22Summer/BodyAtlas/data/QAed/renamed/images'
#txtfile = os.path.join('/nfs/masi/zhouy26/22Summer/BodyAtlas/data/QAed','map_name2.csv')

labeldir = '/nfs/masi/zhouy26/22Summer/BodyAtlas/data/QAed/renamed/labels'

sublist = [s for s in os.listdir(imgdir)]
sublist2 = [s for s in os.listdir(labeldir)]

f = open(txtfile,'w')
writer = csv.writer(f)
writer.writerow(["Old Image ID", "New Image ID", "Old Label ID", "New Image ID"])

for idx in range(len(sublist)): 

    file_name = "Virostko_" + sublist[idx-1].split('_')[1][0:6]
    new_name = os.path.join(imgdir,"img00{}.nii".format(idx))
    old_name = os.path.join(imgdir,sublist[idx-1])


    label_name = os.path.join(labeldir,"img00{}_mask.nii".format(idx))
    mask_name = file_name + "_mask.nii"
    label_old =  os.path.join(labeldir,mask_name)

    data = [sublist[idx-1], "img00{}.nii".format(idx), file_name + "_mask.nii", "img00{}_mask.nii".format(idx)]
    writer.writerow(data)


  

# f.close()