import os
import numpy as np
import torch
from skimage import io
from torch.utils import data
import utils.transform as transform
import matplotlib.pyplot as plt
from skimage.transform import rescale
from torchvision.transforms import functional as F

num_classes = 2
MAS_COLORMAP = [0, 255]
MAS_CLASSES  = ['not road', 'road']

MAS_MEAN = np.array([95.59, 97.08, 89.12])
MAS_STD  = np.array([67.56, 65.90, 68.67])

root = '/data_dir'

def showIMG(img):
    plt.imshow(img)
    plt.show()
    return 0

def normalize_image(im):
    return (im - MAS_MEAN) / MAS_STD

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
    
def get_file_name(mode):
    data_dir = root
    assert mode in ['train', 'val', 'test']
    mask_dir = os.path.join(data_dir, mode, 'sat')
    
    data_list = os.listdir(mask_dir)
    for vi, it in enumerate(data_list):
        data_list[vi] = it[:-5]
    return data_list

def read_RSimages(mode, rescale=False, rotate_aug=False):
    data_dir = root
    assert mode in ['train', 'val', 'test']    
    img_dir = os.path.join(data_dir, mode, 'sat')
    mask_dir = os.path.join(data_dir, mode, 'map')
    
    data_list = os.listdir(img_dir)
    data, labels = [], []
    count=0
    for it in data_list:
        it_name = it[:-5]
        it_ext = it[-5:]
        if (it_ext=='.tiff'):
            img_path = os.path.join(img_dir, it)
            mask_path = os.path.join(mask_dir, it_name+'.tif')
            
            img = io.imread(img_path)
            label = io.imread(mask_path)
            data.append(img)
            labels.append(label[:,:,0])
            count+=1
            if not count%10: print('%d/%d images loaded.'%(count,len(data_list)))
            #if count: break
    print(data[0].shape)
    print(str(len(data)) + ' ' + mode + ' images' + ' loaded.')
    
    return data, labels
    

class RS(data.Dataset):
    def __init__(self, mode, random_crop=False, crop_nums=30, sliding_crop=False, crop_size=640, random_flip=False):
        self.mode = mode
        self.crop_size = crop_size
        self.crop_nums = crop_nums
        self.random_flip = random_flip
        self.random_crop = random_crop
        data, labels = read_RSimages(mode, rescale=False)
        if sliding_crop:
            data, labels = transform.create_crops(data, labels, [self.crop_size, self.crop_size])
            
        self.data = data
        self.labels = Colorls2Index(labels)

        if self.random_crop:
            self.len = crop_nums*len(self.data)
        else:
            self.len = len(self.data)
    
    def __getitem__(self, idx):        
        if self.random_crop:
            idx = int(idx/self.crop_nums)
            data, label = transform.random_crop(self.data[idx], self.labels[idx], size=[self.crop_size, self.crop_size])
        else:                
            data = self.data[idx]
            label = self.labels[idx]
                    
        if self.random_flip:
            data, label = transform.rand_flip(data, label)
        
        data = normalize_image(data)
        data = torch.from_numpy(data.transpose((2, 0, 1)))
        return data, label

    def __len__(self):
        return self.len


