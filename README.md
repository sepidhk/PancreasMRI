# PancreasMRI

## Introduction
The project of PancreasMRI performs pancreas segmentation on abdominal MRI images from UT-Austin using 3D transformer-based model UNesT. The images are T1-weighted MRI from subjects with type I diabetes. Although CT images are widely studied by scientists, very limited research are conducted on the segmentation of abdminal MRI images, and this project is the world's first pancreas segmentation study on MRI images of patients with type I diabetes. In this project, we developed an automated framework to pre-process, train, validation, and test the dataset, and we achieved mean dice 0.724 using five-fold cross validation. 

## Data
The dataset consists of 568 subjects after QAed. They are T1 weighted abdominal MRI images, and all subjects are diagnosed with type I diabetes. The x,y dimension of images are 256 x 256 with 30 - 40 slices. The spacing is 1.5 x 1.5 x 5 mm

## Data Pre-processing 
Because not all subjects have same dimension, and some of the images have 30 slices while others have 40 slices, it's necessary to crop them into the same size to make sure the training is stable. Therefore, we use center crop method to crop all images into dimension (256, 256, 30). In addition, we use linear transformation on all subjects to align their intensity histogram and narrow the display window to (0, 500). 

## Training and Validation

