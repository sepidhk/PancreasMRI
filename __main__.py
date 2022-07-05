# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import nibabel as nb
import scipy.ndimage as ndimage
import numpy as np
import torch
import argparse
from tensorboardX import SummaryWriter
# from apex import amp
from monai.losses import DiceLoss,DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import AsDiscrete,Activations,Compose
from monai.networks.nets import SegResNet
from tqdm import tqdm
from data_utils import get_loader
from networks.unetr import UNETR
# from monai.networks.nets import UNETR
from optimizers.lr_scheduler import WarmupCosineSchedule
#from networks.swin3d_unetrv2 import SwinUNETR as SwinUNETR_v2
from networks.unest import UNesT
from monai.metrics import DiceMetric
from monai.data import decollate_batch

def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000

def Dice(x, y):
    intersect = np.sum(np.sum(np.sum(x * y)))
    y_sum = np.sum(np.sum(np.sum(y)))
    if y_sum == 0:
        return 0.0
    x_sum = np.sum(np.sum(np.sum(x)))
    return 2 * intersect / (x_sum + y_sum)

def resample(img, target_size):
    imx, imy, imz = img.shape
    tx, ty, tz = target_size
    zoom_ratio = ( float(tx) / float(imx), float(ty) / float(imy), float(tz) / float(imz))
    img_resampled = ndimage.zoom( img, zoom_ratio, order=0, prefilter=False)
    return img_resampled

def main():
    def save_ckp(state, checkpoint_dir):
        torch.save(state, checkpoint_dir)

    def train(global_step,train_loader,val_loader,dice_val_best, val_shape_dict):
        model.train()
        epoch_iterator = tqdm(train_loader,desc="Training (X / X Steps) (loss=X.X)",dynamic_ncols=True)
        for step, batch in enumerate(epoch_iterator):
            x, y = (batch["image"].to(device), batch["label"].to(device))
            logit_map = model(x)
            loss = loss_function(logit_map, y)
            if args.amp:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
            optimizer.step()
            if args.lrdecay:
                scheduler.step()
            optimizer.zero_grad()
            epoch_iterator.set_description("Training (%d / %d Steps) (loss=%2.5f)" % (global_step, args.num_steps, loss))
            writer.add_scalar("train/loss", scalar_value=loss, global_step=global_step)

            global_step += 1
            if global_step % args.eval_num == 0 and global_step!=0:
                epoch_iterator_val = tqdm(val_loader, desc="Validate (X / X Steps) (dice=X.X)", dynamic_ncols=True)
                metrics = validation(epoch_iterator_val,val_shape_dict)

                dice_test = metrics
                label_number = 0
                mean_dice = np.mean(dice_test)
                print(dice_test)

                for label_dice in dice_test:
                    writer.add_scalar("Validation/label %d" % (label_number), scalar_value = label_dice, global_step=global_step)
                    label_number += 1


                writer.add_scalar("Validation/Mean Dice BTCV", scalar_value= mean_dice, global_step=global_step)
                # writer.add_scalar("train/loss", scalar_value=loss, global_step=global_step)

                if mean_dice > dice_val_best:
                    checkpoint = {'global_step': global_step, 'state_dict': model.state_dict(),
                                    'optimizer': optimizer.state_dict()}
                    save_ckp(checkpoint, logdir + '/model.pt')
                    dice_val_best = mean_dice
                    print('Model Was Saved ! Current Best Mean Dice: {} Current Best Dice: {} Current Dice: {}'.format(mean_dice, dice_val_best, dice_test))
                else:
                    print('Model Was NOT Saved ! Current Best Mean Dice: {} Current Best Dice: {} Current Dice: {}'.format(mean_dice, dice_val_best,dice_test))
        return global_step, dice_val_best

    def validation(epoch_iterator_val, val_shape_dict):
        model.eval()
        metric_values = []
        roi_size = (args.roi_x, args.roi_y, args.roi_z)
        sw_batch_size = args.sw_batch_size
        with torch.no_grad():
            for step, batch in enumerate(epoch_iterator_val):
                val_inputs, val_labels = (batch["image"].to(device), batch["label"].to(device))
                name = batch["image_meta_dict"]['filename_or_obj'][0].split('/')[-1]
                # target_shape = val_shape_dict[name]
                # print('target shape:{}'.format(target_shape))
                
                val_outputs = sliding_window_inference(val_inputs, roi_size, sw_batch_size, model, overlap=0.2)

                #sigmoid
                # val_outputs = torch.sigmoid(val_outputs)
                # threshold = torch.tensor([0.5]).to(device)
                # val_outputs = (val_outputs > threshold).float()*1
                # val_outputs = (torch.sigmoid(val_outputs) > 0.5).float()

                # val_outputs = val_outputs.cpu().numpy()

                # val_labels = val_labels.float().cpu().numpy()[:,0,:,:,:]


                val_outputs = torch.softmax(val_outputs, 1).cpu().numpy()
                val_outputs = np.argmax(val_outputs, axis = 1).astype(np.uint8)
                val_labels = val_labels.cpu().numpy()[:,0,:,:,:]

                
                dice_list_sub = []
                for i in range(1, args.num_classes):
                    organ_Dice = Dice(val_outputs == i, val_labels == i)
                    dice_list_sub.append(organ_Dice)

                dice_mean = np.mean(dice_list_sub)
                metric_values.append(dice_list_sub)
                epoch_iterator_val.set_description("Validate (%d / %d Steps) (dice_mean=%2.5f)" % (global_step, 10.0, dice_mean, ))
        
        val_metric = []
        for i in range(1, args.num_classes):
            avg_cls = np.mean([l[i-1] for l in metric_values])
            val_metric.append(avg_cls)

        return val_metric

    parser = argparse.ArgumentParser(description='UNETR Training')
    parser.add_argument('--logdir', default=None,type=str)
    parser.add_argument('--pos_embedd', default='perceptron', type=str)
    parser.add_argument('--norm_name', default='instance', type=str)
    parser.add_argument('--gpu', default='0', type=str)
    parser.add_argument('--num_steps', default=20000, type=int)
    parser.add_argument('--eval_num', default=500, type=int)
    parser.add_argument('--warmup_steps', default=500, type=int)
    parser.add_argument('--num_heads', default=16, type=int)
    parser.add_argument('--mlp_dim', default=3072, type=int)
    parser.add_argument('--hidden_size', default=768, type=int)
    parser.add_argument('--feature_size', default=16, type=int)
    parser.add_argument('--in_channels', default=1, type=int)
    parser.add_argument('--out_channels', default=2, type=int)
    parser.add_argument('--num_classes', default=2, type=int)
    parser.add_argument('--res_block', action='store_true')
    parser.add_argument('--conv_block', action='store_true')
    parser.add_argument('--featResBlock', action='store_true')
    parser.add_argument('--roi_x', default=48, type=int)
    parser.add_argument('--roi_y', default=48, type=int)
    parser.add_argument('--roi_z', default=48, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--dropout_rate', default=0.0, type=float)
    parser.add_argument('--fold', default=0, type=int)
    parser.add_argument('--sw_batch_size', default=4, type=int)
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--decay', default=1e-5, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--lrdecay', action='store_true')
    parser.add_argument('--clara_split', action='store_true')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument('--amp_scale', action='store_true')
    parser.add_argument('--max_grad_norm', default=1.0, type=float)
    parser.add_argument('--val', action='store_true')
    parser.add_argument('--model_type', default='nest_unetr', type=str)
    parser.add_argument('--opt_level', default='O2', type=str)
    parser.add_argument('--loss_type', default='dice_ce', type=str)
    parser.add_argument('--opt', default='adamw', type=str)
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--use_pretrained', action='store_true')

    args = parser.parse_args()
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    if args.model_type == 'unetr':
        model = UNETR(
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            img_size=(48,48,48),
            feature_size=args.feature_size,
            hidden_size=args.hidden_size,
            mlp_dim=args.mlp_dim,
            num_heads=args.num_heads,
            pos_embed=args.pos_embedd,
            norm_name=args.norm_name,
            conv_block=args.conv_block,
            res_block=args.res_block,
            dropout_rate=0.0).to(device)
    elif args.model_type == 'segresnet':
        model = SegResNet(
            spatial_dims=3,
            init_filters=16,
            in_channels=4,
            out_channels=3
        ).to(device)
    elif args.model_type == 'swin_unetrv2':

        model = SwinUNETR_v2(in_channels=1,
                          out_channels=14,
                          img_size=(96, 96, 96),
                          feature_size=48,
                          patch_size=2,
                          depths=[2, 2, 2, 2],
                          num_heads=[3, 6, 12, 24],
                          window_size=[7, 7, 7]).to(device)
        
        if args.use_pretrained:
            pretrained_add = '/workspace/Ali/MSD/Task07Pancreas_metric_v830/pretrained_models/model_swinvit.pt'
            model.load_from(weights=torch.load(pretrained_add))
            print('Use pretrained ViT weights from: {}'.format(pretrained_add))
    elif args.model_type == 'nest_unetr':

        model = NestUNETR(in_channels=1,
                          out_channels=14,
                        ).to(device)

    elif args.model_type == 'unest':
        model = UNesT(in_channels=1,
                    out_channels=2,
                ).to(device)                    
            
    # load pre-trained weights
    if args.pretrain:
        if args.ngc:
            args.pretrained_dir = '/ViT-Pytorch/pretrained_models/model_MTK1-3_dataset1-7_156000.pt'
        else:
            args.pretrained_dir = './pretrained_models/'+args.model_type+'.npz'

        model.load_from(weights=torch.load(args.pretrained_dir))
        print('Use pretrained weights')
    model.to(device)
    num_params = count_parameters(model)
    print("Total Parameter: \t%2.1fM" % num_params)
    
    logdir = args.logdir
    writer = SummaryWriter(logdir=logdir)

    # dice_metric = DiceMetric(include_background=True, reduction="none", get_not_nans=False)
    # post_label = AsDiscrete(to_onehot=True, n_classes=args.num_classes)
    # post_pred = AsDiscrete(argmax=True, to_onehot=True, n_classes=args.num_classes)

    if args.opt == "adam":
        optimizer = torch.optim.Adam(params = model.parameters(), lr=args.lr,weight_decay= args.decay)

    elif args.opt == "adamw":
        optimizer = torch.optim.AdamW(params = model.parameters(), lr=args.lr, weight_decay=args.decay)

    elif args.opt == "sgd":
        optimizer = torch.optim.SGD(params = model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.decay)

    if args.amp:
        model, optimizer = amp.initialize(models=model,optimizers=optimizer,opt_level=args.opt_level)
        if args.amp_scale:
            amp._amp_state.loss_scalers[0]._loss_scale = 2 ** 20

    if args.lrdecay:
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=args.num_steps)

    loss_function = DiceCELoss(to_onehot_y=True, squared_pred=True)
    loss_function = DiceCELoss(to_onehot_y=True, softmax=True, squared_pred=True, smooth_nr=0, smooth_dr=1e-6)
    # loss_function = DiceCELoss(to_onehot_y=False, sigmoid=True, softmax=False, squared_pred=True, smooth_nr=0, smooth_dr=1e-6)
    post_trans = Compose([Activations(sigmoid=True), AsDiscrete(threshold_values=True)])

    

    train_loader, val_loader, val_shape_dict, test_loader, test_shape_dict = get_loader(args)
    global_step = 0
    dice_val_best = 0.0


    while global_step < args.num_steps:
        global_step, dice_val_best = train(global_step,train_loader, val_loader, dice_val_best, val_shape_dict)
    checkpoint = {'global_step': global_step,'state_dict': model.state_dict(),'optimizer': optimizer.state_dict()}
    
    save_ckp(checkpoint, logdir+'/model_final_epoch.pt')


if __name__ == '__main__':
    main()
