#!/usr/bin/env bash

FOLD=0
VERSION=v7_crop
BATCH_SIZE=3
MODEL=unest
LOGDIR="./runs/fold${FOLD}_20.${VERSION}"

CUDA_LAUNCH_BLOCKING=1 python __main__.py --fold=${FOLD} --lr=1e-4 --logdir=${LOGDIR} --batch_size=${BATCH_SIZE} --loss_type=dice_ce --opt=adamw \
   --model_type=${MODEL} --eval_num=500 --num_steps=50000 --lrdecay --conv_block --res_block


#python __main__.py --fold=${FOLD} --lr=1e-4 --logdir=${LOGDIR} --batch_size=2 --loss_type=dice_ce --opt=adamw --model_type=swin_unetrv2 --eval_num=50 --num_steps=50000 --lrdecay --ngc --conv_block --res_block --use_pretrained --featResBlock

   

 

