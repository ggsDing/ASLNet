import os
import time
import random
import datetime
import numpy as np
from math import sqrt
import torch.autograd
from skimage import io
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torchvision.utils as vutils
from utils import initialize_weights
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import torchvision.transforms as standard_transforms

working_path = os.path.dirname(os.path.abspath(__file__))

#######################################################
from datasets import Building_Inria as RS
#ED-FCN
from models.ED_FCN import ED_FCN as Seg_Net
#ED-FCN with shape regularization modules
#from models.FCN_SR import FCN_SR as Seg_Net
from models.discriminator import FCDiscriminator as D_Net
NET_NAME = 'ED_FCN_adv'
DATA_NAME = 'INRIA'
#######################################################

from utils.loss import CrossEntropyLoss2d, weighted_BCE_logits
from utils.utils import binary_accuracy as accuracy
from utils.utils import intersectionAndUnion, AverageMeter
   
args = {
    'train_batch_size': 8,
    'val_batch_size': 1,
    'train_crop_size': 512,
    'num_crops': 150,
    'val_crop_size': 2560,
    'lr': 1e-3,
    'lr_D': 5e-4,
    'epochs': 200,
    'gpu': True,
    'weight_decay': 5e-4,
    'momentum': 0.9,
    'print_freq': 200,
    'pred_dir': os.path.join(working_path, 'results', DATA_NAME),
    'chkpt_dir': os.path.join(working_path, 'checkpoints', DATA_NAME),
    'log_dir': os.path.join(working_path, 'logs', DATA_NAME, NET_NAME),
    'load_path': os.path.join(working_path, 'checkpoints', DATA_NAME, 'xxx.pth')
}

if not os.path.exists(args['log_dir']): os.makedirs(args['log_dir'])
if not os.path.exists(args['chkpt_dir']): os.makedirs(args['chkpt_dir'])
if not os.path.exists(args['pred_dir']): os.makedirs(args['pred_dir'])


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def adjust_lr(optimizer, i_iter, all_iter):
    lr = lr_poly(args['lr'], i_iter, all_iter, 0.95)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_lr_D(optimizer, i_iter, all_iter):
    lr = lr_poly(args['lr_D'], i_iter, all_iter, 0.95)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def soft_argmax(seg_map):
    assert seg_map.dim()==4
    # alpha is here to make the largest element really big, so it
    # would become very close to 1 after softmax
    alpha = 1000.0
    b,c,h,w, = seg_map.shape
    soft_max = F.softmax(seg_map*alpha,dim=1)
    return soft_max

def main():        
    model_D = D_Net(1).cuda()
    model_D.train()    
    model_seg = Seg_Net(in_channels=3, num_classes=1).cuda()
        
    train_set = RS.RS('train', load_data=True, random_crop=True, crop_nums=args['num_crops'], crop_size=args['train_crop_size'], random_flip=True) #Inria
    train_loader = DataLoader(train_set, batch_size=args['train_batch_size'], num_workers=4, shuffle=True)
    val_set = RS.RS('val', load_data=True, sliding_crop=True, crop_size=args['val_crop_size']) #Inria
    val_loader = DataLoader(val_set, batch_size=args['val_batch_size'], num_workers=4, shuffle=False)
 
    optimizer = optim.Adam(model_seg.parameters(), lr=args['lr'], betas=(0.9, 0.99))
    optimizer.zero_grad()
        
    optimizer_D = optim.Adam(model_D.parameters(), lr=args['lr_D'], betas=(0.9, 0.99))
    optimizer_D.zero_grad()
    
    bce_loss = torch.nn.BCEWithLogitsLoss()
    MSE_loss = torch.nn.MSELoss().cuda()
    criterion = CrossEntropyLoss2d().cuda()
    # labels for adversarial training
    pred_label = 0
    GT_label = 1
    
    iters_per_epoch = len(train_loader)
    all_iters = iters_per_epoch*args['epochs']
        
    adv_loss_meter = AverageMeter()
    seg_loss_meter = AverageMeter()
    Dpred_loss_meter = AverageMeter()
    Dgt_loss_meter = AverageMeter()
    bestF=0.0
    bestacc=0.0
    bestIoU=0.0
        
    for curr_epoch in range(args['epochs']):
        model_seg.train()
        for i_iter, data in enumerate(train_loader):
            imgs, labels = data
            curr_iter = i_iter + curr_epoch*iters_per_epoch
            imgs = imgs.cuda().float()
            labels = labels.cuda().float().unsqueeze(1)
                    
            optimizer.zero_grad()
            optimizer_D.zero_grad()
            adjust_lr(optimizer, curr_iter, all_iters)
            adjust_lr_D(optimizer_D, curr_iter, all_iters)
            
            ### train G
            # freeze D
            for param in model_D.parameters():
                param.requires_grad = False
                        
            out = model_seg(imgs)
            out_bn = F.sigmoid(out) #soft_argmax(out)
            #aux_bn = F.sigmoid(aux)
            
            loss_seg = MSE_loss(out_bn, labels) *5
            D_out_pred = model_D(out_bn)
            D_out_GT = model_D(labels)
            loss_adv = MSE_loss(F.sigmoid(D_out_pred), F.sigmoid(D_out_GT))
            loss = loss_seg + loss_adv
            loss.backward()
            optimizer.step()
            seg_loss_meter.update(loss_seg.cpu().detach().numpy())
            adv_loss_meter.update(loss_adv.cpu().detach().numpy())
               
            ### train D
            # unfreeze D
            for param in model_D.parameters():
                param.requires_grad = True
            
            # train D with prediction map
            D_out = model_D(out_bn.detach())
            loss_D = bce_loss(D_out, torch.FloatTensor(D_out.data.size()).fill_(pred_label).cuda())
            loss_D.backward()
            Dpred_loss_meter.update(loss_D.cpu().detach().numpy())
                        
            # train D with GT map
            D_out = model_D(labels)
            loss_D = bce_loss(D_out, torch.FloatTensor(D_out.data.size()).fill_(GT_label).cuda())
            loss_D.backward()
            Dgt_loss_meter.update(loss_D.cpu().detach().numpy())
            optimizer_D.step()
        
            if i_iter%args['print_freq'] == 0:
               print('Iter %d/%d, Seg_loss: %.2f, adv_loss: %.2f, Dpred_loss: %.2f, Dgt_loss: %.2f'
                   %(i_iter, iters_per_epoch, seg_loss_meter.val, adv_loss_meter.val, Dpred_loss_meter.val, Dgt_loss_meter.val)) #aux_loss: %.2f, aux_loss_meter.val,
        
        curr_epoch += 1
        val_F, val_acc, val_IoU, val_loss = validate(val_loader, model_seg) #, padding_rate=8
        if val_F>bestF:
            bestF=val_F
            bestacc=val_acc
            bestIoU=val_IoU
            torch.save(model_seg.state_dict(), os.path.join(args['chkpt_dir'], NET_NAME+'_e%d_OA%.2f_F%.2f_IoU%.2f.pth'%(curr_epoch, val_acc*100, val_F*100, val_IoU*100)))
        print('[epoch %d] [lr %f] [Val loss: %.2f Acc %.2f F1 score: %.2f IoU %.2f]' % (curr_epoch, optimizer.param_groups[0]['lr'], val_loss, val_acc*100, val_F*100, val_IoU*100))
        print('Total time: %.1fs, Best rec: Val Acc %.2f F %.2f' %(time.time()-start, bestacc*100, bestF*100))

    print('Training finished.')

def loss_calc(outputs, labels):
    criterion = torch.nn.BCEWithLogitsLoss()
    #criterion = CrossEntropyLoss2d().cuda()
    return criterion(outputs, labels)

def validate(val_loader, model_seg, padding_rate=False, save_pred=True):
    # the following code is written assuming that batch size is 1
    model_seg.eval()
    val_loss = AverageMeter()
    F1_meter = AverageMeter()
    IoU_meter = AverageMeter()
    Acc_meter = AverageMeter()
    
    for vi, data in enumerate(val_loader):
        imgs, labels = data
        imgs = imgs.cuda().float()
        if padding_rate:
            padding_pix = imgs.size()[-1]%padding_rate /2
            padding_pix = [np.floor(padding_pix).astype(int), np.ceil(padding_pix).astype(int), np.floor(padding_pix).astype(int), np.ceil(padding_pix).astype(int)]
            padding_layer = nn.ReflectionPad2d(padding_pix).cuda()
            imgs = padding_layer(imgs)       
        labels = labels.cuda().float().unsqueeze(1)
        with torch.no_grad():
            out = model_seg(imgs)
            if padding_rate: out = out[:,:, padding_pix[0]:-padding_pix[1], padding_pix[2]:-padding_pix[3]]
            loss = loss_calc(out, labels)
            out_bn = F.sigmoid(out)
        val_loss.update(loss.cpu().detach().numpy())

        preds = out_bn.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()
        for (pred, label) in zip(preds, labels):
            acc, precision, recall, F1, IoU = accuracy(pred, label)
            F1_meter.update(F1)
            Acc_meter.update(acc)
            IoU_meter.update(IoU)
        if save_pred and vi==0:
            pred_color = RS.Index2Color(preds[0].squeeze())
            io.imsave(os.path.join(args['pred_dir'], NET_NAME+'.png'), pred_color)
            print('Prediction saved!')

    return F1_meter.avg, Acc_meter.avg, IoU_meter.avg, val_loss.avg

if __name__ == '__main__':
    start = time.time()
    main()
