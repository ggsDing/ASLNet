import os
import cv2
import torch
import numpy as np
from skimage import io
from torch.utils import data
#from osgeo import gdal_array
import matplotlib.pyplot as plt
import utils.transform as transform
from skimage.transform import rescale
from torchvision.transforms import functional as F

num_classes = 2
COLORMAP = [0, 255]
CLASSES  = ['not building', 'building']

MEAN = np.array([94.87, 96.53, 98.60])
STD  = np.array([57.70, 52.40, 50.09])

root = '/data_dir'

def showIMG(img):
    plt.imshow(img)
    plt.show()
    return 0

def normalize_image(im):
    return (im - MEAN) / STD

def normalize_images(imgs):
    for i, im in enumerate(imgs):
        imgs[i] = normalize_image(im)
    return imgs

def Colorls2Index(ColorLabels):
    IndexLabels = []
    for i, data in enumerate(ColorLabels):
        IndexMap = Color2Index(data)
        IndexLabels.append(IndexMap)
    return IndexLabels

def Color2Index(ColorLabel):
    IndexMap = ColorLabel.clip(max=1)
    if len(IndexMap.shape)>2:
        IndexMap = IndexMap[:,:,0]
    return IndexMap

def Index2Color(pred):
    pred = pred*255
    pred = np.asarray(pred, dtype='uint8')
    return pred

def rescale_images(imgs, scale, order):
    for i, im in enumerate(imgs):
        imgs[i] = rescale_image(im, scale, order)
    return imgs
    
def rescale_image(img, scale=1/8, order = 0):
    flag = cv2.INTER_NEAREST
    if order==1: flag = cv2.INTER_LINEAR
    elif order==2: flag = cv2.INTER_AREA
    elif order>2:  flag = cv2.INTER_CUBIC
    im_rescaled = cv2.resize(img, (int(img.shape[0]*scale), int(img.shape[1]*scale)),
                             interpolation=flag)
    return im_rescaled

def get_file_name(mode):
    data_dir = root
    assert mode in ['train', 'test']
    mask_dir = os.path.join(data_dir, mode, 'Images')
    
    data_list = os.listdir(mask_dir)
    for vi, it in enumerate(data_list):
        data_list[vi] = it[:-4]
    return data_list

def read_RSimages(mode, rescale=False, rotate_aug=False):
    data_dir = root
    assert mode in ['train', 'test']    
    img_dir = os.path.join(data_dir, mode, 'Images')
    mask_dir = os.path.join(data_dir, mode, 'GT')
    
    data_list = os.listdir(img_dir)
    data, labels = [], []
    count=0
    for it in data_list:
        it_name = it[:-4]
        it_ext = it[-4:]
        if (it_ext=='.tif'):
            img_path = os.path.join(img_dir, it)
            mask_path = os.path.join(mask_dir, it_name+'.png')
            
            img = io.imread(img_path)
            #label = gdal_array.LoadFile(mask_path)
            label = io.imread(mask_path)
            data.append(img)
            labels.append(label)
            count+=1
            if not count%500: print('%d/%d images loaded.'%(count,len(data_list)))
            #if count: break
    print(data[0].shape)
    print(str(len(data)) + ' ' + mode + ' images' + ' loaded.')
    
    return data, labels
    

class RS(data.Dataset):
    def __init__(self, mode, random_flip=False):
        self.mode = mode
        self.random_flip = random_flip
        data, labels = read_RSimages(mode, rescale=False)
            
        self.data = data
        self.labels = Colorls2Index(labels)

        self.len = len(self.data)
    
    def __getitem__(self, idx):        
        data = self.data[idx]
        label = self.labels[idx]        
        if self.random_flip:
            data, label = transform.rand_flip(data, label)
        
        data = normalize_image(data)
        data = torch.from_numpy(data.transpose((2, 0, 1)))
        return data, label

    def __len__(self):
        return self.len
        