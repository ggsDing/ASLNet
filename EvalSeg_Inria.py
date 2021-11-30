import os
import time
import torch
import numpy as np
import torch.autograd
from skimage import io
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from utils.transform import depadding
########################################
from models.FCN_SR import FCN_SR as Net
from datasets import Building_Inria as RS
DATA_NAME = 'INRIA'
########################################

from utils.loss import CrossEntropyLoss2d
from utils.utils import binary_accuracy as accuracy
from utils.utils import intersectionAndUnion, AverageMeter, CaclTP

working_path = os.path.dirname(os.path.abspath(__file__))
args = {
    'gpu': True,
    'batch_size': 1,
    'val_crop_size': 2560,
    'net_name': 'ASLNet',
    'load_path': os.path.join(working_path, 'checkpoints', DATA_NAME, 'xxx.pth')
}

def soft_argmax(seg_map):
    assert seg_map.dim()==4
    # alpha is here to make the largest element really big, so it
    # would become very close to 1 after softmax
    alpha = 1000.0 
    b,c,h,w, = seg_map.shape
    soft_max = F.softmax(seg_map*alpha,dim=1)
    return soft_max  

def main():
    net = Net(3, num_classes=1).cuda()
    net.load_state_dict(torch.load(args['load_path']))#, strict = False
    net.eval()
    print('Model loaded.')
    pred_path = os.path.join(RS.root, 'pred', args['net_name'])
    if not os.path.exists(pred_path): os.makedirs(pred_path)
    info_txt_path = os.path.join(pred_path, 'info.txt')
    f = open(info_txt_path, 'w+')
        
    pred_name_list = RS.get_file_name('val')
    test_set = RS.RS('val', load_data=True, sliding_crop=True, crop_size=args['val_crop_size'])
    test_loader = DataLoader(test_set, batch_size=args['batch_size'], num_workers=4, shuffle=False)
    predict(net, test_loader, pred_path, pred_name_list, f)

def predict(net, pred_loader, pred_path, pred_name_list, f_out=None, padding_rate=False):
    output_info = f_out is not None
    
    acc_meter = AverageMeter()
    precision_meter = AverageMeter()
    recall_meter = AverageMeter()
    F1_meter = AverageMeter()
    IoU_meter = AverageMeter()
    
    total_iter = len(pred_loader)
    num_files = len(pred_name_list)
    crop_nums = int(total_iter/num_files)
    for vi, data in enumerate(pred_loader):
        imgs, labels = data
        imgs = imgs.cuda().float()
        if padding_rate:
            padding_pix = imgs.size()[-1]%padding_rate /2
            padding_pix = [np.floor(padding_pix).astype(int), np.ceil(padding_pix).astype(int), np.floor(padding_pix).astype(int), np.ceil(padding_pix).astype(int)]
            padding_layer = nn.ReflectionPad2d(padding_pix).cuda()
            imgs = padding_layer(imgs)
        with torch.no_grad(): 
            outputs, _ = net(imgs)
            if padding_rate: outputs = outputs[:,:, padding_pix[0]:-padding_pix[1], padding_pix[2]:-padding_pix[3]] 
            outputs = F.sigmoid(outputs)
        #outputs = soft_argmax(outputs)[:,1,:,:]
        outputs = outputs.detach().cpu().numpy()
        #_, pred = torch.max(output, dim=1)
        for i in range(args['batch_size']):
            idx = vi*args['batch_size']+i
            file_idx = int(idx/crop_nums)
            crop_idx = idx%crop_nums
            if (idx>=total_iter): break
            pred = outputs[i]
            label = labels[i].detach().cpu().numpy()
            acc, precision, recall, F1, IoU = accuracy(pred, label)
            acc_meter.update(acc)
            precision_meter.update(precision)
            recall_meter.update(recall)
            F1_meter.update(F1)
            IoU_meter.update(IoU)
            pred_color = RS.Index2Color(pred.squeeze())
            if crop_nums>1: pred_name = os.path.join(pred_path, pred_name_list[file_idx]+'_%d.png'%crop_idx)
            else: pred_name = os.path.join(pred_path, pred_name_list[file_idx]+'.png')
            io.imsave(pred_name, pred_color)
            
            print('Eval num %d/%d, Acc %.2f, precision %.2f, recall %.2f, F1 %.2f, IoU %.2f'%(idx, total_iter, acc*100, precision*100, recall*100, F1*100, IoU*100))
            if output_info:
                f_out.write('Eval num %d/%d, Acc %.2f, precision %.2f, recall %.2f, F1 %.2f, IoU %.2f\n'%(idx, total_iter, acc*100, precision*100, recall*100, F1*100, IoU*100))
    
    print('avg Acc %.2f, Pre %.2f, Recall %.2f, F1 %.2f, IOU %.2f'%(acc_meter.avg*100, precision_meter.avg*100, recall_meter.avg*100, F1_meter.avg*100, IoU_meter.avg*100))
    
    if output_info:
        f_out.write('Acc %.2f\n'%(acc_meter.avg*100))
        f_out.write('Avg Precision %.2f\n'%(precision_meter.avg*100))
        f_out.write('Avg Recall %.2f\n'%(recall_meter.avg*100))
        f_out.write('Avg F1 %.2f\n'%(F1_meter.avg*100))
        f_out.write('mIoU %.2f\n'%(IoU_meter.avg*100))
    return F1_meter.avg


import torch.nn as nn
class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

if __name__ == '__main__':
    main()
