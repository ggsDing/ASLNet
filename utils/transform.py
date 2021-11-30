import math
import random
import numpy as np
#random.seed(0)
#np.random.seed(0)
import cv2
import skimage
from skimage import transform as sktransf
from skimage.util import pad
import matplotlib.pyplot as plt

def showIMG(img):
    plt.imshow(img)
    plt.show()
    return 0

def rand_flip(img, label):
    r = random.random()
    # showIMG(img.transpose((1, 2, 0)))
    if r < 0.25:
        return img, label
    elif r < 0.5:
        return np.flip(img, axis=0).copy(), np.flip(label, axis=0).copy()
    elif r < 0.75:
        return np.flip(img, axis=1).copy(), np.flip(label, axis=1).copy()
    else:
        return img[::-1, ::-1, :].copy(), label[::-1, ::-1].copy()

def rand_flip3(img, label, label1):
    r = random.random()
    # showIMG(img.transpose((1, 2, 0)))
    if r < 0.25:
        return img, label, label1
    elif r < 0.5:
        return np.flip(img, axis=0).copy(), np.flip(label, axis=0).copy(), np.flip(label1, axis=0).copy()
    elif r < 0.75:
        return np.flip(img, axis=1).copy(), np.flip(label, axis=1).copy(), np.flip(label1, axis=1).copy()
    else:
        return img[::-1, ::-1, :].copy(), label[::-1, ::-1].copy(), label1[::-1, ::-1].copy()

def rand_flip4(img, label, label1, label2):
    r = random.random()
    # showIMG(img.transpose((1, 2, 0)))
    if r < 0.25:
        return img, label, label1, label2
    elif r < 0.5:
        return np.flip(img, axis=0).copy(), np.flip(label, axis=0).copy(), np.flip(label1, axis=0).copy(), np.flip(label2, axis=0).copy()
    elif r < 0.75:
        return np.flip(img, axis=1).copy(), np.flip(label, axis=1).copy(), np.flip(label1, axis=1).copy(), np.flip(label2, axis=1).copy()
    else:
        return img[::-1, ::-1, :].copy(), label[::-1, ::-1].copy(), label1[::-1, ::-1].copy(), label2[::-1, ::-1].copy()

def rand_flip_mix(img, label, x_s):
    r = random.random()
    # showIMG(img.transpose((1, 2, 0)))
    if r < 0.25:
        return img, label, x_s
    elif r < 0.5:
        return np.flip(img, axis=0).copy(), np.flip(label, axis=0).copy(), np.flip(x_s, axis=0).copy()
    elif r < 0.75:
        return np.flip(img, axis=1).copy(), np.flip(label, axis=1).copy(), np.flip(x_s, axis=1).copy()
    else:
        return img[::-1, ::-1, :].copy(), label[::-1, ::-1].copy(), x_s[::-1, ::-1, :].copy()

def rand_rotate(img, label):
    r = random.randint(0,179)
    # print(r)
    # showIMG(img.transpose((1, 2, 0)))
    img_rotate = np.asarray(sktransf.rotate(img, r, order=1, mode='symmetric',
                                            preserve_range=True), np.float)
    label_rotate = np.asarray(sktransf.rotate(label, r, order=0, mode='constant',
                                               cval=0, preserve_range=True), np.uint8)
    # print(img_rotate[0:10, 0:10, :])
    # print(label_rotate[0:10, 0:10])
    # h_s = image
    return img_rotate, label_rotate

def rand_rotate_crop(img, label):
    r = random.randint(0,179)
    image_height, image_width = img.shape[0:2]
    im_rotated = rotate_image(img, r, order=1)
    l_rotated = rotate_image(label, r, order=0)
    crop_w, crop_h = largest_rotated_rect(image_width, image_height, math.radians(r))
    im_rotated_cropped = crop_around_center(im_rotated, crop_w, crop_h)
    l_rotated_cropped = crop_around_center(l_rotated, crop_w, crop_h)
    # print(img_rotate[0:10, 0:10, :])
    # print(label_rotate[0:10, 0:10])
    # h_s = image
    return im_rotated_cropped, l_rotated_cropped

def rotate_image(image, angle, order=0):
    """
    Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
    (in degrees). The returned image will be large enough to hold the entire
    new image, with a black background
    """

    # Get the image size
    # No that's not an error - NumPy stores image matricies backwards
    image_size = (image.shape[1], image.shape[0])
    image_center = tuple(np.array(image_size) / 2)

    # Convert the OpenCV 3x2 rotation matrix to 3x3
    rot_mat = np.vstack(
        [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
    )

    rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

    # Shorthand for below calcs
    image_w2 = image_size[0] * 0.5
    image_h2 = image_size[1] * 0.5

    # Obtain the rotated coordinates of the image corners
    rotated_coords = [
        (np.array([-image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2,  image_h2]) * rot_mat_notranslate).A[0],
        (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
        (np.array([ image_w2, -image_h2]) * rot_mat_notranslate).A[0]
    ]

    # Find the size of the new image
    x_coords = [pt[0] for pt in rotated_coords]
    x_pos = [x for x in x_coords if x > 0]
    x_neg = [x for x in x_coords if x < 0]

    y_coords = [pt[1] for pt in rotated_coords]
    y_pos = [y for y in y_coords if y > 0]
    y_neg = [y for y in y_coords if y < 0]

    right_bound = max(x_pos)
    left_bound = min(x_neg)
    top_bound = max(y_pos)
    bot_bound = min(y_neg)

    new_w = int(abs(right_bound - left_bound))
    new_h = int(abs(top_bound - bot_bound))

    # We require a translation matrix to keep the image centred
    trans_mat = np.matrix([
        [1, 0, int(new_w * 0.5 - image_w2)],
        [0, 1, int(new_h * 0.5 - image_h2)],
        [0, 0, 1]
    ])

    # Compute the tranform for the combined rotation and translation
    affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

    # Apply the transform
    flag = cv2.INTER_NEAREST
    if order == 1: flag = cv2.INTER_LINEAR
    elif order == 2: flag = cv2.INTER_AREA
    elif order > 2: flag = cv2.INTER_CUBIC

    result = cv2.warpAffine(
        image,
        affine_mat,
        (new_w, new_h),
        flags=flag
    )

    return result

def rand_rotate_mix(img, label, x_s):
    r = random.randint(0,179)
    # print(r)
    # showIMG(img.transpose((1, 2, 0)))
    img_rotate = np.asarray(sktransf.rotate(img, r, order=1, mode='symmetric',
                                            preserve_range=True), np.float)
    label_rotate = np.asarray(sktransf.rotate(label, r, order=0, mode='constant',
                                               cval=0, preserve_range=True), np.uint8)
    x_s_rotate = np.asarray(sktransf.rotate(x_s, r, order=0, mode='symmetric',
                                               cval=0, preserve_range=True), np.uint8)
    # print(img_rotate[0:10, 0:10, :])
    # print(label_rotate[0:10, 0:10])
    # h_s = image
    return img_rotate, label_rotate, x_s_rotate

def create_crops(ims, labels, size):
    crop_imgs = []
    crop_labels = []
    label_dims = len(labels[0].shape)
    for img, label in zip(ims, labels):
        h = img.shape[0]
        w = img.shape[1]
        c_h = size[0]
        c_w = size[1]
        if h < c_h or w < c_w:
            print("Cannot crop area {} from image with size ({}, {})".format(str(size), h, w))
            crop_imgs.append(img)
            crop_labels.append(label)
            continue
        h_rate = h/c_h
        w_rate = w/c_w
        h_times = math.ceil(h_rate)
        w_times = math.ceil(w_rate)
        if h_times==1: stride_h=0
        else:
            stride_h = math.ceil(c_h*(h_times-h_rate)/(h_times-1))            
        if w_times==1: stride_w=0
        else:
            stride_w = math.ceil(c_w*(w_times-w_rate)/(w_times-1))
        for j in range(h_times):
            for i in range(w_times):
                s_h = int(j*c_h - j*stride_h)
                if(j==(h_times-1)): s_h = h - c_h
                e_h = s_h + c_h
                s_w = int(i*c_w - i*stride_w)
                if(i==(w_times-1)): s_w = w - c_w
                e_w = s_w + c_w
                # print('%d %d %d %d'%(s_h, e_h, s_w, e_w))
                # print('%d %d %d %d'%(s_h_s, e_h_s, s_w_s, e_w_s))
                crop_imgs.append(img[s_h:e_h, s_w:e_w, :])
                if label_dims==2:
                    crop_labels.append(label[s_h:e_h, s_w:e_w])
                else:
                    crop_labels.append(label[s_h:e_h, s_w:e_w, :])

    print('Sliding crop finished. %d images created.' %len(crop_imgs))
    return crop_imgs, crop_labels
    
def create_crops_onlyimgs(ims, size):
    crop_imgs = []
    for img in ims:
        h = img.shape[0]
        w = img.shape[1]
        c_h = size[0]
        c_w = size[1]
        if h < c_h or w < c_w:
            print("Cannot crop area {} from image with size ({}, {})".format(str(size), h, w))
            continue
        h_rate = h/c_h
        w_rate = w/c_w
        h_times = math.ceil(h_rate)
        w_times = math.ceil(w_rate)
        stride_h = math.ceil(c_h*(h_times-h_rate)/(h_times-1))
        stride_w = math.ceil(c_w*(w_times-w_rate)/(w_times-1))
        for j in range(h_times):
            for i in range(w_times):
                s_h = int(j*c_h - j*stride_h)
                if(j==(h_times-1)): s_h = h - c_h
                e_h = s_h + c_h
                s_w = int(i*c_w - i*stride_w)
                if(i==(w_times-1)): s_w = w - c_w
                e_w = s_w + c_w
                # print('%d %d %d %d'%(s_h, e_h, s_w, e_w))
                # print('%d %d %d %d'%(s_h_s, e_h_s, s_w_s, e_w_s))
                crop_imgs.append(img[s_h:e_h, s_w:e_w, :])

    print('Sliding crop finished. %d images created.' %len(crop_imgs))
    return crop_imgs

def create_crops3(ims, labels, labels1, size):
    crop_imgs = []
    crop_labels = []
    crop_labels1 = []
    crop_labels2 = []
    label_dims = len(labels[0].shape)
    for img, label, label1 in zip(ims, labels, labels1):
        h = img.shape[0]
        w = img.shape[1]
        c_h = size[0]
        c_w = size[1]
        if h < c_h or w < c_w:
            print("Cannot crop area {} from image with size ({}, {})".format(str(size), h, w))
            crop_imgs.append(img)
            crop_labels.append(label)
            crop_labels1.append(label1)
            continue
        h_rate = h/c_h
        w_rate = w/c_w
        h_times = math.ceil(h_rate)
        w_times = math.ceil(w_rate)
        if h_times==1: stride_h=0
        else:
            stride_h = math.ceil(c_h*(h_times-h_rate)/(h_times-1))            
        if w_times==1: stride_w=0
        else:
            stride_w = math.ceil(c_w*(w_times-w_rate)/(w_times-1))
        for j in range(h_times):
            for i in range(w_times):
                s_h = int(j*c_h - j*stride_h)
                if(j==(h_times-1)): s_h = h - c_h
                e_h = s_h + c_h
                s_w = int(i*c_w - i*stride_w)
                if(i==(w_times-1)): s_w = w - c_w
                e_w = s_w + c_w
                # print('%d %d %d %d'%(s_h, e_h, s_w, e_w))
                # print('%d %d %d %d'%(s_h_s, e_h_s, s_w_s, e_w_s))
                crop_imgs.append(img[s_h:e_h, s_w:e_w, :])
                if label_dims==2:
                    crop_labels.append(label[s_h:e_h, s_w:e_w])
                    crop_labels1.append(label1[s_h:e_h, s_w:e_w])
                else:
                    crop_labels.append(label[s_h:e_h, s_w:e_w, :])
                    crop_labels1.append(label1[s_h:e_h, s_w:e_w, :])

    print('Sliding crop finished. %d images created.' %len(crop_imgs))
    return crop_imgs, crop_labels, crop_labels1

def create_crops4(ims, labels, labels1, labels2, size):
    crop_imgs = []
    crop_labels = []
    crop_labels1 = []
    crop_labels2 = []
    label_dims = len(labels[0].shape)
    for img, label, label1, label2 in zip(ims, labels, labels1, labels2):
        h = img.shape[0]
        w = img.shape[1]
        c_h = size[0]
        c_w = size[1]
        if h < c_h or w < c_w:
            print("Cannot crop area {} from image with size ({}, {})".format(str(size), h, w))
            crop_imgs.append(img)
            crop_labels.append(label)
            crop_labels1.append(label1)
            crop_labels2.append(label2)
            continue
        h_rate = h/c_h
        w_rate = w/c_w
        h_times = math.ceil(h_rate)
        w_times = math.ceil(w_rate)
        if h_times==1: stride_h=0
        else:
            stride_h = math.ceil(c_h*(h_times-h_rate)/(h_times-1))            
        if w_times==1: stride_w=0
        else:
            stride_w = math.ceil(c_w*(w_times-w_rate)/(w_times-1))
        for j in range(h_times):
            for i in range(w_times):
                s_h = int(j*c_h - j*stride_h)
                if(j==(h_times-1)): s_h = h - c_h
                e_h = s_h + c_h
                s_w = int(i*c_w - i*stride_w)
                if(i==(w_times-1)): s_w = w - c_w
                e_w = s_w + c_w
                # print('%d %d %d %d'%(s_h, e_h, s_w, e_w))
                # print('%d %d %d %d'%(s_h_s, e_h_s, s_w_s, e_w_s))
                crop_imgs.append(img[s_h:e_h, s_w:e_w, :])
                if label_dims==2:
                    crop_labels.append(label[s_h:e_h, s_w:e_w])
                    crop_labels1.append(label1[s_h:e_h, s_w:e_w])
                    crop_labels2.append(label2[s_h:e_h, s_w:e_w])
                else:
                    crop_labels.append(label[s_h:e_h, s_w:e_w, :])
                    crop_labels1.append(label1[s_h:e_h, s_w:e_w, :])
                    crop_labels2.append(label2[s_h:e_h, s_w:e_w, :])

    print('Sliding crop finished. %d images created.' %len(crop_imgs))
    return crop_imgs, crop_labels, crop_labels1, crop_labels2

def center_crop(ims, labels, size):
    crop_imgs = []
    crop_labels = []
    for img, label in zip(ims, labels):
        h = img.shape[0]
        w = img.shape[1]
        c_h = size[0]
        c_w = size[1]
        if h < c_h or w < c_w:
            print("Cannot crop area {} from image with size ({}, {})".format(str(size), h, w))
            continue
        s_h = int(h/2 - c_h/2)
        e_h = s_h + c_h
        s_w = int(w/2 - c_w/2)
        e_w = s_w + c_w
        crop_imgs.append(img[s_h:e_h, s_w:e_w, :])
        crop_labels.append(label[s_h:e_h, s_w:e_w, :])

    print('Center crop finished. %d images created.' %len(crop_imgs))
    return crop_imgs, crop_labels

def five_crop(ims, labels, size):
    crop_imgs = []
    crop_labels = []
    for img, label in zip(ims, labels):
        h = img.shape[0]
        w = img.shape[1]
        c_h = size[0]
        c_w = size[1]
        if h < c_h or w < c_w:
            print("Cannot crop area {} from image with size ({}, {})".format(str(size), h, w))
            continue
        s_h = int(h/2 - c_h/2)
        e_h = s_h + c_h
        s_w = int(w/2 - c_w/2)
        e_w = s_w + c_w
        crop_imgs.append(img[s_h:e_h, s_w:e_w, :])
        crop_labels.append(label[s_h:e_h, s_w:e_w, :])

        crop_imgs.append(img[0:c_h, 0:c_w, :])
        crop_labels.append(label[0:c_h, 0:c_w, :])
        crop_imgs.append(img[h-c_h:h, w-c_w:w, :])
        crop_labels.append(label[h-c_h:h, w-c_w:w, :])
        crop_imgs.append(img[0:c_h, w-c_w:w, :])
        crop_labels.append(label[0:c_h, w-c_w:w, :])
        crop_imgs.append(img[h-c_h:h, 0:c_w, :])
        crop_labels.append(label[h-c_h:h, 0:c_w, :])

    print('Five crop finished. %d images created.' %len(crop_imgs))
    return crop_imgs, crop_labels

def img_padding(img, scale=8):
    shape_before = img.shape
    img_dims = len(shape_before)
    h, w = img.shape[:2]
    h_padding = h%scale
    w_padding = w%scale
    need_padding = h_padding or w_padding
    if need_padding:
        h_padding = (scale-h_padding)/2
        h_padding1 = math.ceil(h_padding)
        h_padding2 = math.floor(h_padding)
        w_padding = (scale-w_padding)/2
        w_padding1 = math.ceil(w_padding)
        w_padding2 = math.floor(w_padding)
        if img_dims==2:
            img = pad(img, ((h_padding1, h_padding2), (w_padding1, w_padding2)), 'constant')
        else:
            img = pad(img, ((h_padding1, h_padding2), (w_padding1, w_padding2), (0,0)), 'symmetric')
        shape_after = img.shape
    return img
    

def align_dims(np_input, expected_dims=2):
    dim_input = len(np_input.shape)
    np_output = np_input
    if dim_input>expected_dims:
        np_output = np_input.squeeze(0)
    elif dim_input<expected_dims:
        np_output = np.expand_dims(np_input, 0)        
    assert len(np_output.shape) == expected_dims
    return np_output

def depadding(pred, target, padding_info):
    pred = align_dims(pred, 2)
    target = align_dims(target, 2)
    h, w = pred.shape
    h_padding1, h_padding2, w_padding1, w_padding2 = padding_info
    pred = pred[h_padding1:h-h_padding2, w_padding1:w-w_padding2]
    target = target[h_padding1:h-h_padding2, w_padding1:w-w_padding2]
    return pred, target

def data_depadding(preds, targets, padding_data):
    for idx, padding_info in enumerate(padding_data):
        preds[idx], targets[idx] = depadding(preds[idx], targets[idx], padding_info)
    return preds, targets

def data_padding(imgs, labels, scale=32, return_data=False):
    label_dims = len(labels[0].shape)
    padding_data = []
    for idx, img in enumerate(imgs):
        label = labels[idx]
        shape_before = img.shape
        h, w = img.shape[:2]
        h_padding = h%scale
        w_padding = w%scale
        need_padding = h_padding or w_padding
        if need_padding:
            h_padding = (scale-h_padding)/2
            h_padding1 = math.ceil(h_padding)
            h_padding2 = math.floor(h_padding)
            
            w_padding = (scale-w_padding)/2
            w_padding1 = math.ceil(w_padding)
            w_padding2 = math.floor(w_padding)
            img = pad(img, ((h_padding1, h_padding2), (w_padding1, w_padding2), (0,0)), 'symmetric')            
            if label_dims==2:
                label = pad(label, ((h_padding1, h_padding2), (w_padding1, w_padding2)), 'constant')
            else:
                label = pad(label, ((h_padding1, h_padding2), (w_padding1, w_padding2), (0,0)), 'constant')
            padding_data.append([h_padding1, h_padding2, w_padding1, w_padding2])
            shape_after = img.shape
            #print('img padding: [%d, %d]->[%d, %d]'%(shape_before[0],shape_before[1],shape_after[0],shape_after[1]))
            imgs[idx] = img
            labels[idx] = label
    print('Image padding finished.')
    if return_data:
        return imgs, labels, padding_data
    else:
        return imgs, labels

def data_padding3(imgs, labels, labels1, scale=32, return_data=False):
    label_dims = len(labels[0].shape)
    padding_data = []
    for idx, img in enumerate(imgs):
        label = labels[idx]
        label1 = labels1[idx]
        shape_before = img.shape
        h, w = img.shape[:2]
        h_padding = h%scale
        w_padding = w%scale
        need_padding = h_padding or w_padding
        if need_padding:
            h_padding = (scale-h_padding)/2
            h_padding1 = math.ceil(h_padding)
            h_padding2 = math.floor(h_padding)
            
            w_padding = (scale-w_padding)/2
            w_padding1 = math.ceil(w_padding)
            w_padding2 = math.floor(w_padding)
            img = pad(img, ((h_padding1, h_padding2), (w_padding1, w_padding2), (0,0)), 'symmetric')            
            if label_dims==2:
                label = pad(label, ((h_padding1, h_padding2), (w_padding1, w_padding2)), 'constant')
                label1 = pad(label1, ((h_padding1, h_padding2), (w_padding1, w_padding2)), 'symmetric')
            else:
                label = pad(label, ((h_padding1, h_padding2), (w_padding1, w_padding2), (0,0)), 'constant')
                label1 = pad(label1, ((h_padding1, h_padding2), (w_padding1, w_padding2), (0,0)), 'symmetric')
            padding_data.append([h_padding1, h_padding2, w_padding1, w_padding2])
            shape_after = img.shape
            #print('img padding: [%d, %d]->[%d, %d]'%(shape_before[0],shape_before[1],shape_after[0],shape_after[1]))
            imgs[idx] = img
            labels[idx] = label
            labels1[idx] = label1
    print('Image padding finished.')
    if return_data:
        return imgs, labels, labels1, padding_data
    else:
        return imgs, labels, labels1

def data_padding4(imgs, labels, labels1, labels2, scale=32, return_data=False):
    label_dims = len(labels[0].shape)
    padding_data = []
    for idx, img in enumerate(imgs):
        label = labels[idx]
        label1 = labels1[idx]
        label2 = labels2[idx]
        shape_before = img.shape
        h, w = img.shape[:2]
        h_padding = h%scale
        w_padding = w%scale
        need_padding = h_padding or w_padding
        if need_padding:
            h_padding = (scale-h_padding)/2
            h_padding1 = math.ceil(h_padding)
            h_padding2 = math.floor(h_padding)
            
            w_padding = (scale-w_padding)/2
            w_padding1 = math.ceil(w_padding)
            w_padding2 = math.floor(w_padding)
            img = pad(img, ((h_padding1, h_padding2), (w_padding1, w_padding2), (0,0)), 'symmetric')            
            if label_dims==2:
                label = pad(label, ((h_padding1, h_padding2), (w_padding1, w_padding2)), 'constant')
                label1 = pad(label1, ((h_padding1, h_padding2), (w_padding1, w_padding2)), 'symmetric')
                label2 = pad(label2, ((h_padding1, h_padding2), (w_padding1, w_padding2)), 'symmetric')
            else:
                label = pad(label, ((h_padding1, h_padding2), (w_padding1, w_padding2), (0,0)), 'constant')
                label1 = pad(label1, ((h_padding1, h_padding2), (w_padding1, w_padding2), (0,0)), 'symmetric')
                label2 = pad(label2, ((h_padding1, h_padding2), (w_padding1, w_padding2), (0,0)), 'symmetric')
            padding_data.append([h_padding1, h_padding2, w_padding1, w_padding2])
            shape_after = img.shape
            #print('img padding: [%d, %d]->[%d, %d]'%(shape_before[0],shape_before[1],shape_after[0],shape_after[1]))
            imgs[idx] = img
            labels[idx] = label
            labels1[idx] = label1
            labels2[idx] = label2
    print('Image padding finished.')
    if return_data:
        return imgs, labels, labels1, labels2, padding_data
    else:
        return imgs, labels, labels1, labels2

def five_crop_mix(ims, labels, x_s, size, scale=8):
    crop_imgs = []
    crop_labels = []
    crop_xs = []
    for img, label, x_s in zip(ims, labels, x_s):
        h = img.shape[0]
        w = img.shape[1]
        h_s = int(h/scale)
        w_s = int(w/scale)
        c_h = size[0]
        c_w = size[1]
        c_h_s = int(c_h/scale)
        c_w_s = int(c_w/scale)
        if h < c_h or w < c_w:
            print("Cannot crop area {} from image with size ({}, {})".format(str(size), h, w))
            continue
        s_h_s = int(h_s/2 - c_h_s/2)
        e_h_s = s_h_s + c_h_s
        s_w_s = int(w_s/2 - c_w_s/2)
        e_w_s = s_w_s + c_w_s
        s_h = s_h_s*scale
        s_w = s_w_s*scale
        e_h = s_h+c_h
        e_w = s_w+c_w
        
        crop_xs.append(x_s[:, s_h_s:e_h_s, s_w_s:e_w_s])
        crop_imgs.append(img[s_h:e_h, s_w:e_w, :])
        crop_labels.append(label[s_h:e_h, s_w:e_w, :])

        crop_xs.append(x_s[:, :c_h_s, :c_w_s])
        crop_imgs.append(img[:c_h, :c_w, :])
        crop_labels.append(label[:c_h, :c_w, :])
        
        crop_xs.append(x_s[:, -c_h_s:, -c_w_s:])
        crop_imgs.append(img[-c_h:, -c_w:, :])
        crop_labels.append(label[-c_h:, -c_w:, :])
        
        crop_xs.append(x_s[:, :c_h_s, -c_w_s:])
        crop_imgs.append(img[:c_h, -c_w:, :])
        crop_labels.append(label[:c_h, -c_w:, :])
        
        crop_xs.append(x_s[:, -c_h_s:, :c_w_s])
        crop_imgs.append(img[-c_h:, :c_w, :])
        crop_labels.append(label[-c_h:, :c_w, :])

    print('Five crop finished. %d images created.' %len(crop_imgs))
    return crop_imgs, crop_labels, crop_xs

def sliding_crop(img, size):
    # print(img.shape)
    h = img.shape[0]
    w = img.shape[1]
    c_h = size[0]
    c_w = size[1]
    if h < c_h or w < c_w:
        print("Cannot crop area {} from image with size ({}, {})"
              .format(str(size), h, w))
    else:
        h_rate = h/c_h
        w_rate = w/c_w
        h_times = math.ceil(h_rate)
        w_times = math.ceil(w_rate)
        stride_h = math.ceil(c_h*(h_times-h_rate)/(h_times-1))
        stride_w = math.ceil(c_w*(w_times-w_rate)/(w_times-1))
        crop_imgs = []
        for j in range(h_times):
            for i in range(w_times):
                s_h = int(j*c_h - j*stride_h)
                if(j==(h_times-1)): s_h = h - c_h
                e_h = s_h + c_h
                s_w = int(i*c_w - i*stride_w)
                if(i==(w_times-1)): s_w = w - c_w
                e_w = s_w + c_w
                # print('%d %d %d %d'%(s_h, e_h, s_w, e_w))
                crop_im = img[s_h:e_h, s_w:e_w, :]
                crop_imgs.append(crop_im)

                # crop_imgs_f = []
                # for im in crop_imgs:
                #     crop_imgs_f.append(cv2.flip(im, -1))

                # crops = np.concatenate((np.array(crop_imgs)), axis=0)
                # print(crops.shape)
        return crop_imgs

def random_crop(img, label, size):
    # print(img.shape)
    h = img.shape[0]
    w = img.shape[1]
    c_h = size[0]
    c_w = size[1]
    if h < c_h or w < c_w:
        print("Cannot crop area {} from image with size ({}, {})"
              .format(str(size), h, w))
    else:
        s_h = random.randint(0, h-c_h)
        e_h = s_h + c_h
        s_w = random.randint(0, w-c_w)
        e_w = s_w + c_w

        crop_im = img[s_h:e_h, s_w:e_w, :]
        crop_label = label[s_h:e_h, s_w:e_w]
        # print('%d %d %d %d'%(s_h, e_h, s_w, e_w))
        return crop_im, crop_label

def random_crop3(img, label, label1, size):
    # print(img.shape)
    h = img.shape[0]
    w = img.shape[1]
    c_h = size[0]
    c_w = size[1]
    if h < c_h or w < c_w:
        print("Cannot crop area {} from image with size ({}, {})"
              .format(str(size), h, w))
    else:
        s_h = random.randint(0, h-c_h)
        e_h = s_h + c_h
        s_w = random.randint(0, w-c_w)
        e_w = s_w + c_w

        crop_im = img[s_h:e_h, s_w:e_w, :]
        crop_label = label[s_h:e_h, s_w:e_w]
        crop_label1 = label1[s_h:e_h, s_w:e_w]
        # print('%d %d %d %d'%(s_h, e_h, s_w, e_w))
        return crop_im, crop_label, crop_label1

def random_crop4(img, label, label1, label2, size):
    # print(img.shape)
    h = img.shape[0]
    w = img.shape[1]
    c_h = size[0]
    c_w = size[1]
    if h < c_h or w < c_w:
        print("Cannot crop area {} from image with size ({}, {})"
              .format(str(size), h, w))
    else:
        s_h = random.randint(0, h-c_h)
        e_h = s_h + c_h
        s_w = random.randint(0, w-c_w)
        e_w = s_w + c_w

        crop_im = img[s_h:e_h, s_w:e_w, :]
        crop_label = label[s_h:e_h, s_w:e_w]
        crop_label1 = label1[s_h:e_h, s_w:e_w]
        crop_label2 = label2[s_h:e_h, s_w:e_w]
        # print('%d %d %d %d'%(s_h, e_h, s_w, e_w))
        return crop_im, crop_label, crop_label1, crop_label2

def random_crop_mix(img, label, x_s, size, scale=8):
    # print(img.shape)
    h = img.shape[0]
    w = img.shape[1]
    c_h = size[0]
    c_w = size[1]
    c_h_s = int(c_h/scale)
    c_w_s = int(c_w/scale)
    h_times = int(h/scale - c_h_s)
    w_times = int(w/scale - c_w_s)
    if h < c_h or w < c_w:
        print("Cannot crop area {} from image with size ({}, {})"
              .format(str(size), h, w))
    else:
        s_h_s = random.randint(0, h_times)
        s_h = s_h_s * scale
        s_w_s = random.randint(0, w_times)
        s_w = s_w_s * scale
        e_h_s = s_h_s + c_h_s
        e_w_s = s_w_s + c_w_s
        e_h = s_h + c_h
        e_w = s_w + c_w

        crop_im = img[s_h:e_h, s_w:e_w, :]
        crop_label = label[s_h:e_h, s_w:e_w]
        crop_xs = x_s[:, s_h_s:e_h_s, s_w_s:e_w_s]
        # print('%d %d %d %d' % (s_h, e_h, s_w, e_w))
        # print('%d %d %d %d' % (s_h_s, e_h_s, s_w_s, e_w_s))
        return crop_im, crop_label, crop_xs

def create_crops_mix(ims, labels, x_s, size, scale=1/8):
    crop_imgs = []
    crop_labels = []
    crop_x_s = []
    for img, label, x in zip(ims, labels, x_s):
        h = img.shape[0]
        w = img.shape[1]
        c_h = size[0]
        c_w = size[1]
        c_h_s = int(c_h*scale)
        c_w_s = int(c_w*scale)
        if h < c_h or w < c_w:
            print("Cannot crop area {} from image with size ({}, {})".format(str(size), h, w))
            continue
        h_rate = h/c_h
        w_rate = w/c_w
        h_times = math.ceil(h_rate)
        w_times = math.ceil(w_rate)
        stride_h = math.ceil(c_h*(h_times-h_rate)/(h_times-1))
        stride_w = math.ceil(c_w*(w_times-w_rate)/(w_times-1))
        for j in range(h_times):
            for i in range(w_times):
                s_h = int(j*c_h - j*stride_h)
                s_h_s = int(s_h*scale)
                if(j==(h_times-1)): s_h = h - c_h
                e_h = s_h + c_h
                e_h_s = s_h_s + c_h_s
                s_w = int(i*c_w - i*stride_w)
                s_w_s = int(s_w*scale)
                if(i==(w_times-1)): s_w = w - c_w
                e_w = s_w + c_w
                e_w_s = s_w_s + c_w_s
                crop_imgs.append(img[s_h:e_h, s_w:e_w, :])
                crop_labels.append(label[s_h:e_h, s_w:e_w, :])
                crop_x_s.append(x[:, s_h_s:e_h_s, s_w_s:e_w_s])

    print('Sliding crop finished. %d images created.' %len(crop_imgs))
    return crop_imgs, crop_labels, crop_x_s

def crop_around_center(image, width, height):
    """
    Given a NumPy / OpenCV 2 image, crops it to the given width and height,
    around it's centre point
    """

    image_size = (image.shape[1], image.shape[0])
    image_center = (int(image_size[0] * 0.5), int(image_size[1] * 0.5))

    if(width > image_size[0]):
        width = image_size[0]

    if(height > image_size[1]):
        height = image_size[1]

    x1 = int(image_center[0] - width * 0.5)
    x2 = int(image_center[0] + width * 0.5)
    y1 = int(image_center[1] - height * 0.5)
    y2 = int(image_center[1] + height * 0.5)

    return image[y1:y2, x1:x2]

def largest_rotated_rect(w, h, angle):
    """
    Given a rectangle of size wxh that has been rotated by 'angle' (in
    radians), computes the width and height of the largest possible
    axis-aligned rectangle within the rotated rectangle.

    Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

    Converted to Python by Aaron Snoswell
    """

    quadrant = int(math.floor(angle / (math.pi / 2))) & 3
    sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
    alpha = (sign_alpha % math.pi + math.pi) % math.pi

    bb_w = w * math.cos(alpha) + h * math.sin(alpha)
    bb_h = w * math.sin(alpha) + h * math.cos(alpha)

    gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

    delta = math.pi - alpha - gamma

    length = h if (w < h) else w

    d = length * math.cos(alpha)
    a = d * math.sin(alpha) / math.sin(delta)

    y = a * math.cos(gamma)
    x = y * math.tan(gamma)

    return (
        bb_w - 2 * x,
        bb_h - 2 * y
    )

def Rotate_Aug(imgs, labels, step=20, start_angle=20, max_angle=179):
    for idx in range(len(imgs)):
        im = imgs[idx]
        l = labels[idx]
        image_height, image_width = im.shape[0:2]
        for i in range(start_angle, max_angle, step):
            im_rotated = rotate_image(im, i, order=3)
            l_rotated  = rotate_image(l,  i, order=0)
            crop_w, crop_h = largest_rotated_rect(image_width, image_height, math.radians(i))
            im_rotated_cropped = crop_around_center(im_rotated, crop_w, crop_h)
            l_rotated_cropped = crop_around_center(l_rotated, crop_w, crop_h)
            imgs.append(im_rotated_cropped)
            labels.append(l_rotated_cropped)
        print('Img %d rotated.'%idx)
    print('Rotation finished. %d images in total.'%len(imgs))
    return imgs, labels

def Rotate_Aug_S(im, l, step=20, start_angle=15, max_angle=89):
    imgs = []
    labels = []
    image_height, image_width = im.shape[0:2]
    for i in range(start_angle, max_angle, step):
        im_rotated = rotate_image(im, i, order=1)
        l_rotated  = rotate_image(l,  i, order=0)
        crop_w, crop_h = largest_rotated_rect(image_width, image_height, math.radians(i))
        im_rotated_cropped = crop_around_center(im_rotated, crop_w, crop_h)
        l_rotated_cropped = crop_around_center(l_rotated, crop_w, crop_h)
        imgs.append(im_rotated_cropped)
        labels.append(l_rotated_cropped)
    print('Rotation finished. %d images added.'%len(imgs))
    return imgs, labels

