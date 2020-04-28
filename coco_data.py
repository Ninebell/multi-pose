from conf import *
import os
import json
import random
import math
from PIL import Image, ImageDraw
from scipy.ndimage.filters import gaussian_filter
import numpy as np
from tqdm import tqdm
from skimage.draw import line_aa
import matplotlib.pyplot as plt

file_name_format = "{0:012d}.jpg"


def load_annotaion():
    json_data = open(root_path+val_annotation_path, 'r').read()
    data = json.loads(json_data)
    return data['annotations']


def create_kernel(shape, key_points):
    base = np.zeros(shape)
    return base


def get_image(id, is_train=True):
    path = train_path if is_train else validate_path
    file_path = '{0}{1}{2}{3}'.format(root_path,path,'\\input\\',file_name_format.format(id))
    img = Image.open(file_path)
    return np.array(img)


def change_to_heat_map(key_points, box, org_shape, num_keypoint, shape):
    key = np.asarray(key_points).reshape((-1, 3))
    base = np.zeros((shape[0], shape[1], key.shape[0]+1))
    sgm = np.log(box[2] + box[3])/4

    center_x = 0
    center_y = 0

    for i in range(key.shape[0]):
        x = int(key[i,0])
        y = int(key[i,1])
        if x == y == 0:
            continue
        center_x += x
        center_y += y

        base_x = int(x/org_shape[0]*shape[0])
        base_y = int(y/org_shape[1]*shape[1])

        if base_x >= 64:
            base_x = 63
        if base_y >= 63:
            base_y = 63
        # print(y,x,org_shape,base_x,base_y)
        base[base_y,base_x,i] = 1
        base[:,:,i] = gaussian_filter(base[:,:,i],sigma=sgm)
        base[:,:,i] = base[:,:,i]/np.max(base[:,:,i])

    center_x = int(center_x/num_keypoint/org_shape[0]*shape[0])
    center_y = int(center_y/num_keypoint/org_shape[1]*shape[1])
    if center_y >= 64:
        center_y = 63
    if center_x >= 64:
        center_x = 63

    base[center_y,center_x,key.shape[0]] = 1
    base[:,:,key.shape[0]] = gaussian_filter(base[:,:,key.shape[0]],sigma=sgm)
    base[:,:,key.shape[0]] = base[:, :,key.shape[0]] / np.max(base[:, :,key.shape[0]])

    if (shape[0]*0.1<=center_x<=shape[0]*0.9 and shape[1]*0.1<=center_y<=shape[1]*0.9):
        return base
    return None


def change_to_limb_map(key_points, box, org_shape, num_keypoint, shape):
    key = np.asarray(key_points).reshape((-1, 3))
    base = np.zeros((shape[0], shape[1], key.shape[0]))
    sgm = np.log(box[2] + box[3])/4

    center_x = 0
    center_y = 0

    for i in range(key.shape[0]):
        x = int(key[i,0])
        y = int(key[i,1])
        if x == y == 0:
            continue
        center_x += x
        center_y += y

    center_x = int(center_x/num_keypoint/org_shape[0]*shape[0])
    center_y = int(center_y/num_keypoint/org_shape[1]*shape[1])
    if center_y >= 64:
        center_y = 63
    if center_x >= 64:
        center_x = 63
    if not (shape[0]*0.1<=center_x<=shape[0]*0.9 and shape[1]*0.1<=center_y<=shape[1]*0.9):
        return None

    for i in range(key.shape[0]):
        x = int(key[i, 0])
        y = int(key[i, 1])
        if x == y == 0:
            continue
        base_x = int(x / org_shape[0] * shape[0])
        base_y = int(y / org_shape[1] * shape[1])

        if base_x >= 64:
            base_x = 63
        if base_y >= 63:
            base_y = 63
        rr, cc, val = line_aa(base_y,base_x,center_y,center_x)
        base[rr,cc,i] = val/np.max(val)
        base[:,:,i] = gaussian_filter(base[:,:,i],sigma=sgm)
        base[:,:,i] = base[:,:,i]/np.max(base[:,:,i])

    return base


def heat_map_create(annotation_dict, shape, is_train=True):
    inner = train_path if is_train else validate_path

    os.makedirs(root_path+inner+heat_map_path,exist_ok=True)


    for i in tqdm(range(len(annotation_dict))):
        img_id = annotation_dict[i]['image_id']
        box = annotation_dict[i]['bbox']
        key_points = annotation_dict[i]['keypoints']
        num_key_points = annotation_dict[i]['num_keypoints']

        if num_key_points < 10:
            continue

        img = get_image(img_id, is_train)
        heat_maps = change_to_heat_map(key_points, box, img.shape, num_key_points, shape)

        if heat_maps is None:
            continue

        all = np.zeros(shape)

        for j in range(heat_maps.shape[2]):
            j_label_path = root_path+inner+heat_map_path+"\\{}\\".format(j)
            img_file_path = j_label_path+file_name_format.format(img_id)
            os.makedirs(j_label_path, exist_ok=True)

            heat_map = np.asarray(heat_maps[:, :, j] * 255, dtype=np.uint8)
            all = np.maximum(all, heat_map)
            if os.path.exists(img_file_path):
                org = np.asarray(Image.open(img_file_path))
                heat_map = np.maximum(org,heat_map)

            Image.fromarray(heat_map).save(img_file_path)

        j += 1
        j_label_path = root_path + inner + heat_map_path + "\\{}\\".format(j)
        os.makedirs(j_label_path, exist_ok=True)
        img_file_path = j_label_path + file_name_format.format(img_id)
        all = np.asarray(all, dtype=np.uint8)
        if os.path.exists(img_file_path):
            org = np.asarray(Image.open(img_file_path))
            all = np.maximum(org, all)

        Image.fromarray(all).save(img_file_path)


def center_map_create(annotation_dict, shape, is_train=True):
    inner = train_path if is_train else validate_path

    os.makedirs(root_path+inner+limb_path,exist_ok=True)

    for i in tqdm(range(len(annotation_dict))):
        img_id = annotation_dict[i]['image_id']
        box = annotation_dict[i]['bbox']
        key_points = annotation_dict[i]['keypoints']
        num_key_points = annotation_dict[i]['num_keypoints']
        if num_key_points < 10:
            continue

        img = get_image(img_id,is_train)
        limb_maps = change_to_limb_map(key_points, box, img.shape, num_key_points, shape)

        if limb_maps is None:
            continue

        all = np.zeros(shape)

        for j in range(limb_maps.shape[2]):
            j_label_path = root_path + inner + limb_path+"\\{}\\".format(j)
            img_file_path = j_label_path+file_name_format.format(img_id)
            os.makedirs(j_label_path, exist_ok=True)

            limb_map = np.asarray(limb_maps[:, :, j] * 255, dtype=np.uint8)

            all = np.maximum(all, limb_map)
            if os.path.exists(img_file_path):
                org = np.asarray(Image.open(img_file_path))
                limb_map = np.maximum(org,limb_map)

            Image.fromarray(limb_map).save(img_file_path)
        j += 1
        j_label_path = root_path + inner + limb_path + "\\{}\\".format(j)
        os.makedirs(j_label_path, exist_ok=True)
        img_file_path = j_label_path + file_name_format.format(img_id)
        all = np.asarray(all, dtype=np.uint8)
        if os.path.exists(img_file_path):
            org = np.asarray(Image.open(img_file_path))
            all = np.maximum(org, all)

        Image.fromarray(all).save(img_file_path)


def data_generator(batch_size, shuffle=True, is_train=True):
    base_path = root_path+train_path if is_train else root_path+validate_path
    dirs = os.listdir(base_path+heat_map_path+'0')
    if shuffle:
        random.shuffle(dirs)
    batch_iter = len(dirs)//batch_size
    for idx in range(batch_iter):
        # print(idx)
        x = []
        heat_maps = []
        limbs = []

        for b_i in range(batch_size):
            file_name = dirs[idx*batch_size + b_i].split('.')[0]
            img = Image.open(base_path+'\\input\\'+file_name+".jpg")
            if img.mode == 'L':
                continue
            # print(img.mode, file_name)
            img = img.resize((256,256))

            x.append(np.asarray(img)/255.)

            heat_map_file_path = base_path + heat_map_path + '\\{}\\'.format(0) + file_name + ".jpg"
            base_heat = np.asarray(Image.open(heat_map_file_path)) / 255.
            base_heat = np.reshape(base_heat, (64,64,1))
            for h in range(1, 18):
                heat_map_file_path = base_path+heat_map_path+'\\{}\\'.format(h)+file_name+".jpg"
                heat = np.asarray(Image.open(heat_map_file_path))/255.
                heat = np.reshape(heat, (64,64,1))
                base_heat = np.concatenate([base_heat, heat], axis=-1)

            limb_map_file_path = base_path + limb_path + '\\{}\\'.format(0) + file_name + ".jpg"
            base_limb = np.asarray(Image.open(limb_map_file_path)) / 255.
            base_limb = np.reshape(base_limb, (64,64,1))
            for l in range(1, 17):
                limb_map_file_path = base_path+limb_path+'\\{}\\'.format(l)+file_name+".jpg"
                limb = np.asarray(Image.open(limb_map_file_path))/255.
                limb = np.reshape(limb, (64,64,1))
                base_limb = np.concatenate([base_limb, limb], axis=-1)

            heat_maps.append(base_heat)
            limbs.append(base_limb)

        x = np.asarray(x)
        heat_maps = np.asarray(heat_maps)
        limbs = np.asarray(limbs)
        yield x, [heat_maps, limbs, heat_maps, limbs]


if __name__ == "__main__":
    for data in data_generator(10, is_train=True):
        print(1)
    # shape = (64,64)
    # anno = load_annotaion()
    # heat_map_create(anno, shape, False)
    # center_map_create(anno, shape, False)
