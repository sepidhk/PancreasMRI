# --------------------------------------------------------
# Copyright (C) 2021 NVIDIA Corporation. All rights reserved.
# Nvidia Source Code License-NC
# Transformer Pretraining Code: Yucheng, Vishwesh, Ali
# --------------------------------------------------------
import os
import numpy as np
from numpy.random import randint
from PIL import Image
import nibabel as nb
import json
# Generate JSON file


imgdir = '/nfs/masi/zhouy26/22Summer/BodyAtlas/data/QAed/training/images'
testdir = '/nfs/masi/zhouy26/22Summer/BodyAtlas/data/QAed/testing/images'
 


imglist = [s for s in os.listdir(imgdir)]
testlist = [i for i in os.listdir(testdir)]


# trainpath = os.path.join('./', 'testing')
# testpath = os.path.join('./', 'data')


imgchunks = [imglist[x:x+92] for x in range(0, len(imglist), 92)]


for fold in range(5):
    datadict = {}
    datadict['training'] = []
    datadict['validation'] = []
    datadict['testing'] = []

    for idx, l in enumerate(imgchunks):
        for t in l:
            ifile = "./training/images/" + t
            t_dict = {'image': '', 'label':''}
            t_dict['image'] = ifile
            ilabel = "./training/labels/" + t.split('.nii')[0] + "_mask.nii"
            t_dict['label'] = ilabel
            if idx != fold:
                datadict['training'].append(t_dict)
            else:               
                datadict['validation'].append(t_dict)

    for test in testlist:
        ifile = "./testing/images/" + test
        t_dict = {'image': '', 'label':''}
        t_dict['image'] = ifile
        ilabel ="./testing/labels/" + test.split('.nii')[0] + "_mask.nii"
        t_dict['label'] = ilabel
        datadict['testing'].append(t_dict)

            
    new_jsonfile = '/nfs/masi/zhouy26/22Summer/BodyAtlas/json/fold{}.json'.format(fold) 

    with open(new_jsonfile, 'w') as f:
        json.dump(datadict, f, indent=4, sort_keys=True)
    f.close()    
