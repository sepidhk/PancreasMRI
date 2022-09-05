# PancreasMRI

## Introduction
The project of PancreasMRI performs pancreas segmentation on abdominal MRI images from UT-Austin using 3D transformer-based model UNesT. The images are T1-weighted MRI from subjects with type I diabetes. Although CT images are widely studied by scientists, very limited research are conducted on the segmentation of abdminal MRI images, and this project is the world's first pancreas segmentation study on MRI images of patients with type I diabetes. In this project, we developed an automated framework to pre-process, train, validation, and test the dataset, and we achieved mean dice 0.724 using five-fold cross validation. 

## Data
The dataset consists of 568 subjects after QAed. They are T1 weighted abdominal MRI images, and all subjects are diagnosed with type I diabetes. The x,y dimension of images are 256 x 256 with 30 - 40 slices. The spacing is 1.5 x 1.5 x 5 mm

## Data Pre-processing 
Because not all subjects have same dimension, and some of the images have 30 slices while others have 40 slices, it's necessary to crop them into the same size to make sure the training is stable. Therefore, we use center crop method to crop all images into dimension (256, 256, 30). In addition, we use linear transformation on all subjects to align their intensity histogram and narrow the display window to (0, 500). 

<<<<<<< HEAD
## Model Overview
The segmentation used the newly released model UNesT on MONAI which achieved high performance in medical image segmentation. The 3D transformer structure allows it to capture semantic information and its CNN-based structure enables it to extract local information. The network structure is demonstrated in the diagram below:

## Training and Validation
The experiments used fivefold cross validation method where 368 subjects are used for training, 92 subjects are used for validation, and 108 subjects are used for testing. 3D transfomer model UNesT is used to learn segmentation. The inputs are MRI scans, the manual labels from clinicians, and the outpus will be a predicted pancreas label where the pixel index is 1. The dice metrics are used to evaluate the segmentation, and DiceCE loss is used during the training. Data augmentation, including random flip, random affine, random rotate, random shift intensity are implemented. And the input size is (32, 32, 32) The training and validation curve is as below:


## Results on Testing Set
Using the model saved from fivefold cross validation, it achieves mean dice 0.724 on the testing set, and the example outputs is as below where red label is the predicted label and green label is the truth label.

## Model Overview 
The segmentation used the newly released model UNesT on MONAI which achieved high performance in medical image segmentation. The 3D transformer structure allows it to capture semantic information and its CNN-based structure enables it to extract local information. The network structure is demonstrated in the diagram below:

## Training and Validation
The experiments used fivefold cross validation method where 368 subjects are used for training, 92 subjects are used for validation, and 108 subjects are used for testing. 3D transfomer model UNesT is used to learn segmentation. The inputs are MRI scans, the manual labels from clinicians, and the outpus will be a predicted pancreas label where the pixel index is 1. The dice metrics are used to evaluate the segmentation, and DiceCE loss is used during the training. Data augmentation, including random flip, random affine, random rotate, random shift intensity are implemented. And the input size is (32, 32, 32) The training and validation curve is as below: 

## Results on Testing Set
Using the model saved from fivefold cross validation, it achieves mean dice 0.724 on the testing set, and the example outputs is as below where red label is the predicted label and green label is the truth label. 



