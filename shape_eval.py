import os
import cv2
import time
import numpy as np
from skimage import io, measure
from skimage.color import label2rgb

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, count, weight):
        self.val = val
        self.avg = val
        self.count = count
        self.sum = val * weight
        self.initialized = True

    def update(self, val, count=1, weight=1):
        if not self.initialized:
            self.initialize(val, count, weight)
        else:
            self.add(val, count, weight)

    def add(self, val, count, weight):
        self.val = val
        self.count += count
        self.sum += val * weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg
    
#input: binary image
def bn_region_growing(img, seed, region_limit=False, return_range=True):
    #print('region grow at seed: [%d, %d]'%(seed[0], seed[1]))
    #Parameters for region growing
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    #Input image parameters
    h, w = img.shape
    #Initialize segmented output image
    segmented_obj = np.zeros((h, w), np.uint8)
    loc_range = [seed[0], seed[0], seed[1], seed[1]]
    seed_list = [seed]
    segmented_obj[seed[0], seed[1]] = 1
    
    #Region growing until ...
    perimeter = 0
    while len(seed_list):
        if region_limit:
            if np.sum(segmented_obj)>region_limit: break
        check_seed = seed_list.pop(0)
        #boundary_mark = False
        for offsets in neighbors:
            n_x = check_seed[0] + offsets[0]
            n_y = check_seed[1] + offsets[1]
            if n_x<0 or n_x>=h or n_y<0 or n_y>=w: continue
            #if not img[n_x, n_y]: boundary_mark = True
            if img[n_x, n_y] and segmented_obj[n_x, n_y]==0:
                segmented_obj[n_x, n_y] = 1
                seed_list.append([n_x, n_y])
                if return_range:
                    if n_x<loc_range[0]: loc_range[0]=n_x
                    if n_x>loc_range[1]: loc_range[1]=n_x
                    if n_y<loc_range[2]: loc_range[2]=n_y
                    if n_y>loc_range[3]: loc_range[3]=n_y
        #if boundary_mark: perimeter+=1
    #print('object area: %d, perimeter: %d.'%(np.sum(segmented_obj), perimeter))
    #io.imsave('/home/dinglei/Code/BSeg_pred/binary/obj%d%d.png'%(seed[0], seed[1]), segmented_obj*255)
    if return_range: return segmented_obj, loc_range
    else: return segmented_obj

def get_chain_code(boundary):
    current = boundary[-1][0]
    chain = []
    for i in boundary:
        i = i[0]
        dx = i[0]-current[0]
        dy = i[1]-current[1]
        if dx < 0 and dy == 0:
            chain.append(0)
        if dx < 0 and dy < 0:
            chain.append(1)
        if dx == 0 and dy < 0:
            chain.append(2)
        if dx > 0 and dy < 0:
            chain.append(3)
        if dx > 0 and dy == 0:
            chain.append(4)
        if dx > 0 and dy > 0:
            chain.append(5)
        if dx == 0 and dy > 0:
            chain.append(6)
        if dx < 0 and dy > 0:
            chain.append(7)
        current = i
    return chain
    
def calc_curvature(chain):
    curvature = 0
    current = chain[-1]
    for i in chain:
        dif = np.abs(i - current)
        assert dif<8, "chain code out of range."
        if dif>4: dif = 8-dif
        curvature += dif
    return curvature

def mark_img(img, bn_thred=0):
    img = (img>bn_thred).astype(np.uint8)
    if img.ndim>2: img = img[:,:,0]
    h,w = img.shape
    
    #Parameters for region growing
    img_index = np.zeros((h,w)).astype(np.uint64)
    objects = []
    obj_id = 0
    for i in range(h):
        for j in range(w):
            if img[i,j]>0 and img_index[i,j]==0:
                segmented_obj, loc_range = bn_region_growing(img, [i,j])
                obj_id += 1
                img = img-segmented_obj
                img_index += segmented_obj*obj_id
                obj = seg_object(obj_id, segmented_obj, loc_range)
                if obj.area>15: objects.append(obj)
    print('Index image generated. Num_objects: %d'%len(objects))          
    #rgb_map = label2rgb(img_index)
    return img_index, objects

class seg_object(object):
    def __init__(self, index, segmented_map, loc_range):
        self.idx = index
        self.loc_range = loc_range
        self.area = np.sum(segmented_map)
        if self.area<15: return
        contours, _ = cv2.findContours(segmented_map, 2, 1)
        self.perimeter = cv2.arcLength(contours[0],True)
        chain_code = get_chain_code(contours[0])  
        #total absolute curvature     
        self.curv = calc_curvature(chain_code)/self.perimeter
        if len(contours)>1:
            for i in range(1, len(contours)): self.perimeter += cv2.arcLength(contours[i],True)
        #self.perimeter = measure.perimeter(segmented_map, neighbourhood=4)
        if not self.perimeter: self.perimeter=0.01
        p_eac = np.sqrt(self.area*np.pi)*2
        self.compact = p_eac/self.perimeter
            
    def get_map(self, index_map):
        return (index_map==self.idx).astype(np.uint8)

def shape_eval(index_GT_map, index_pred_map):
    start = time.time()
    index_GT_map, objects_GT = mark_img(GT_img, bn_thred=127)
    index_pred_map, objects_pred = mark_img(pred_img, bn_thred=127)
    
    thred_overseg = 0.7
    thred_underseg = 0.7
    num_match = 0
    compact_meter = AverageMeter()
    curve_meter = AverageMeter()
    for item_GT in objects_GT:
        GT_item_map = item_GT.get_map(index_GT_map)
        h0, h1, w0, w1 = item_GT.loc_range
        #area_match_thred = [int(item_GT.area*0.7), int(item_GT.area*1.3)]
        for item_pred in objects_pred:
            u0, u1, v0, v1 =  item_pred.loc_range
            outbound = False
            if u0>h1 or u1<h0 or v0>w1 or v1<w0:
                outbound = True
            if not outbound:
                pred_item_map = item_pred.get_map(index_pred_map)
                intersection = pred_item_map*GT_item_map
                insct_area = np.sum(intersection)
                r_overseg = insct_area/item_GT.area
                r_underseg = insct_area/item_pred.area
                if r_underseg > thred_underseg and r_overseg > thred_overseg:
                    num_match += 1
                    compact_error = np.abs(item_GT.compact-item_pred.compact)
                    curve_error = np.abs(item_GT.curv-item_pred.curv)
                    compact_meter.update(compact_error)
                    curve_meter.update(curve_error)
                    #print('match item found. compact error: %.2f'%compact_error)
                    continue
    match_ratio = num_match/len(objects_GT)
    print('Running time: %.2f match items: %d. Match rate: %.2f, mean compact error: %.2f curv error: %.2f.'\
        %(time.time()-start, num_match, match_ratio*100, compact_meter.avg*100, curve_meter.avg*100))
    return match_ratio, compact_meter.avg, curve_meter.avg
    

if __name__ == '__main__':
    GT_dir = '/YOUR_GT_DIR/'
    pred_dir = '/YOUR_PRED_DIR/'
    score = []
    data_list = os.listdir(pred_dir)    
    m_meter = AverageMeter()    
    compact_meter = AverageMeter()
    curve_meter = AverageMeter()
    for it in data_list:
        if it[-4:]=='.png':
            pred_path = os.path.join(pred_dir, it)
            pred_img = io.imread(pred_path)
            GT_path = os.path.join(GT_dir, it[:-4]+'.tif')            
            GT_img = io.imread(GT_path)
            print(GT_path)
            
            match_ratio, mcompact_error, mcurve_error = shape_eval(GT_img, pred_img)
            m_meter.update(match_ratio)
            compact_meter.update(mcompact_error)
            curve_meter.update(mcurve_error)
            #if not score.count%10: print('%d images processed, average score %.2f.'%(score.count, score.avg*100))
    print('Average match rate: %.2f, avg compact error: %.2f, avg curv error: %.2f'%(m_meter.avg*100, compact_meter.avg*100, curve_meter.avg*100))
    
