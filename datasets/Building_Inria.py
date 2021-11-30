import os
import cv2
import torch
import numpy as np
from skimage import io, exposure
from torch.utils import data
import matplotlib.pyplot as plt
import utils.transform as transform
from skimage.transform import rescale
from torchvision.transforms import functional as F

num_classes = 2
INRIA_COLORMAP = [0, 255]
INRIA_CLASSES  = ['not building', 'building']

INRIA_MEAN = np.array([103.25, 108.97, 100.16])
INRIA_STD  = np.array([51.35, 46.80, 44.93])
root = '/data_dir'
mean_dict = {'aus': [100.94, 103.53,  97.64],
             'chi': [103.36, 109.49,  98.18],
             'kit': [ 90.78,  98.12,  84.41],
             'wtyr':[107.54, 125.68, 122.08],
             'vie': [122.71, 119.91, 113.43],
             'sfo': [151.41, 156.86, 138.57],
             'bel': [ 96.03,  99.83,  83.28],
             'blo': [113.68, 117.82,  91.25],
             'etyr': [112.68, 122.81, 113.88],
             'inn': [105.55, 106.66,  99.35]}
std_dict = {'aus': [44.32, 43.08, 41.81],
            'chi': [54.49, 53.16, 51.39],            
            'kit': [43.79, 37.38, 34.97],
            'wtyr':[54.55, 49.35, 40.58],
            'vie': [57.56, 50.90, 49.64],
            'sfo': [65.61, 59.42, 60.44],
            'bel': [46.25, 41.91, 41.84],
            'blo': [46.47, 43.77, 41.21],
            'etyr':[46.21, 39.45, 38.60],
            'inn': [42.47, 36.72, 34.05]}

def normalize_image_test(im, img_name):
    area_prefix = img_name[:3]
    if area_prefix=='tyr':
        if img_name[:7]=='tyrol-w': area_prefix='wtyr'
        elif img_name[:7]=='tyrol-e': area_prefix='etyr'
    mean = mean_dict[area_prefix]
    std = std_dict[area_prefix]
    return (im - mean) / std

def normalize_image(im):
    return (im - INRIA_MEAN) / INRIA_STD

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
    return IndexMap

def Index2Color(pred):
    pred = pred*255
    pred = np.asarray(pred, dtype='uint8')
    return pred

def rescale_images(imgs, scale, order):
    for i, im in enumerate(imgs):
        imgs[i] = rescale_image(im, scale, order)
    return imgs

def rescale_image(img, scale=8.0, order = 0):
    flag = cv2.INTER_NEAREST
    if order==1: flag = cv2.INTER_LINEAR
    elif order==2: flag = cv2.INTER_AREA
    elif order>2:  flag = cv2.INTER_CUBIC
    im_rescaled = cv2.resize(img, (int(img.shape[0]/scale), int(img.shape[1]/scale)), interpolation=flag)
    return im_rescaled

def get_file_name(mode):
    data_dir = root
    assert mode in ['train', 'val', 'test']
    mask_dir = os.path.join(data_dir, mode, 'images')
    
    data_list = os.listdir(mask_dir)
    for vi, it in enumerate(data_list):
        data_list[vi] = it[:-4]
    return data_list

def read_RSimages(mode, load_data=False):
    data_dir = root
    assert mode in ['train', 'val', 'test']    
    img_dir = os.path.join(data_dir, mode, 'images')
    mask_dir = os.path.join(data_dir, mode, 'gt')
    
    data_list = os.listdir(img_dir)
    data, labels = [], []
    count=0
    for it in data_list:
        it_name = it[:-4]
        it_ext = it[-4:]
        if (it_ext=='.tif'):
            img_path = os.path.join(img_dir, it)
            label_path = os.path.join(mask_dir, it_name+'.tif')            
            if load_data:
                img = io.imread(img_path)
                label = io.imread(label_path)
                data.append(img)
                labels.append(label)
            else:
                data.append(img_path)
                labels.append(label_path)
            count+=1
            if not count%10: print('%d/%d images loaded.'%(count,len(data_list)))
            #if count>10: break
    if load_data: print(data[0].shape)
    print(str(len(data)) + ' ' + mode + ' images' + ' loaded.')    
    return data, labels

class RS(data.Dataset):
    def __init__(self, mode, load_data=False, random_crop=False, crop_nums=30, sliding_crop=False, crop_size=640, random_flip=False):
        self.mode = mode
        self.load_data = load_data
        self.crop_size = crop_size
        self.crop_nums = crop_nums
        self.random_flip = random_flip
        self.random_crop = random_crop
        data, labels = read_RSimages(mode, load_data)
        if sliding_crop and load_data:
            data, labels = transform.create_crops(data, labels, [self.crop_size, self.crop_size])
        self.data = data
        self.labels = labels

        if self.random_crop:
            self.len = crop_nums*len(self.data)
        else:
            self.len = len(self.data)
    
    def __getitem__(self, idx):
        if self.random_crop:
            idx = int(idx/self.crop_nums)
                
        if self.load_data:
            data = self.data[idx]
            label = self.labels[idx]
        else:
            data = io.imread(self.data[idx])
            label = io.imread(self.labels[idx])
                
        if self.random_crop:
            data, label = transform.random_crop(data, label, size=[self.crop_size, self.crop_size])                    
        if self.random_flip:
            data, label = transform.rand_flip(data, label)
        
        data = normalize_image(data)
        data = torch.from_numpy(data.transpose((2, 0, 1)))
        
        return data, Color2Index(label)

    def __len__(self):
        return self.len

class RS_test(data.Dataset):
    def __init__(self, input_path, crop_size):
        data_list = os.listdir(input_path)
        data = []
        mask_name_list = []
        data_length = int(len(data_list))
        
        count = 0
        for it in data_list:
            if (it[-4:]=='.tif'):
                mask_name = it[:-4] + '.png'
                mask_name_list.append(mask_name)
                img_path = os.path.join(input_path, it)            
                img = io.imread(img_path)
                data.append(img) 
                count +=1
                if not count%10: print('%d/%d images loaded.'%(count,len(data_list)))
        if crop_size:
            data = transform.create_crops_onlyimgs(data, [crop_size, crop_size])
        self.data = data
        self.mask_name_list = mask_name_list
        self.len = len(self.data)
        self.crop_nums = self.len // len(self.mask_name_list)
        print('crom_nums: %d'%self.crop_nums)
    
    def get_mask_name(self, idx):
        return self.mask_name_list[idx]
    
    def __getitem__(self, idx):
        idx_file = idx//self.crop_nums
        data = normalize_image_test(self.data[idx], self.mask_name_list[idx_file])
        data = torch.from_numpy(data.transpose((2, 0, 1)))
        return data

    def __len__(self):
        return self.len

