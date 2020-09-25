from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from scipy.io import loadmat
from PIL import Image, ImageDraw
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import math
import json
from conf import *

from conf import *

import os
import cv2
import random

kernel_size = 5
base_shape = (64,64,1)
base_shape_2 = (64,64,2)
base_shape_3 = (64,64,3)


def decode(image):
    return np.asarray(image * 255, dtype=np.uint8)


def encode(image):
    return np.asarray(image, dtype=np.float)/255.


def get_test_set():

    # joint_data_fn = 'dataset/mpii/data.json'
    mat = loadmat('E:\\dataset\\mpii\\mpii_human_pose_v1_u12_1.mat')

    file_name = []

    for i, (anno, train_flag) in enumerate(
        zip(mat['RELEASE']['annolist'][0, 0][0],
            mat['RELEASE']['img_train'][0, 0][0])):

        img_fn = anno['image']['name'][0, 0][0]
        train_flag = int(train_flag)

        if train_flag == 0:
            file_name.append(img_fn)

    return file_name


def save_joints():
    joint_data_fn = 'dataset/mpii/data.json'
    # mat = loadmat('E:\\dataset\\mpii\\mpii_human_pose_v1_u12_1.mat')
    mat = loadmat('C:\\Users\\rnwhd\\Desktop\\git\\multi-pose\\dataset\\mpii\\mpii_human_pose_v1_u12_1.mat')

    fp = open(joint_data_fn, 'w')

    for i, (anno, train_flag) in enumerate(
        zip(mat['RELEASE']['annolist'][0, 0][0],
            mat['RELEASE']['img_train'][0, 0][0])):

        img_fn = anno['image']['name'][0, 0][0]
        train_flag = int(train_flag)

        head_rect = []
        if 'x1' in str(anno['annorect'].dtype):
            head_rect = zip(
                [x1[0, 0] for x1 in anno['annorect']['x1'][0]],
                [y1[0, 0] for y1 in anno['annorect']['y1'][0]],
                [x2[0, 0] for x2 in anno['annorect']['x2'][0]],
                [y2[0, 0] for y2 in anno['annorect']['y2'][0]])

        if 'annopoints' in str(anno['annorect'].dtype):
            print(train_flag)
            if train_flag == 0:
                print('h')
            annopoints = anno['annorect']['annopoints'][0]
            head_x1s = anno['annorect']['x1'][0]
            head_y1s = anno['annorect']['y1'][0]
            head_x2s = anno['annorect']['x2'][0]
            head_y2s = anno['annorect']['y2'][0]
            for annopoint, head_x1, head_y1, head_x2, head_y2 in zip(
                    annopoints, head_x1s, head_y1s, head_x2s, head_y2s):
                if len(annopoint) != 0:
                    head_rect = [float(head_x1[0, 0]),
                                 float(head_y1[0, 0]),
                                 float(head_x2[0, 0]),
                                 float(head_y2[0, 0])]

                    # joint coordinates
                    annopoint = annopoint['point'][0, 0]
                    j_id = [str(j_i[0, 0]) for j_i in annopoint['id'][0]]
                    x = [x[0, 0] for x in annopoint['x'][0]]
                    y = [y[0, 0] for y in annopoint['y'][0]]
                    joint_pos = {}
                    for _j_id, (_x, _y) in zip(j_id, zip(x, y)):
                        joint_pos[str(_j_id)] = [float(_x), float(_y)]
                    # joint_pos = fix_wrong_joints(joint_pos)

                    # visiblity list
                    if 'is_visible' in str(annopoint.dtype):
                        vis = [v[0] if v else [0]
                               for v in annopoint['is_visible'][0]]
                        vis = dict([(k, int(v[0])) if len(v) > 0 else v
                                    for k, v in zip(j_id, vis)])
                    else:
                        vis = None

                    if len(joint_pos) == 16:
                        data = {
                            'filename': 'E:\\dataset\\mpii\\mpii_human_pose_v1\\images\\'+img_fn,
                            'train': train_flag,
                            'head_rect': head_rect,
                            'is_visible': vis,
                            'joint_pos': joint_pos
                        }

                        # print(json.dumps(data), file=fp)


def write_line(datum, fp):
    joints = sorted([[int(k), v] for k, v in datum['joint_pos'].items()])
    joints = np.array([j for i, j in joints]).flatten()

    out = [datum['filename']]
    out.extend(joints)
    out = [str(o) for o in out]
    out = ','.join(out)

    print(out, file=fp)


def split_train_test():
    fp_test = open('dataset/mpii/test_joints.csv', 'w')
    fp_train = open('dataset/mpii/train_joints.csv', 'w')
    all_data = open('dataset/mpii/data.json').readlines()
    N = len(all_data)
    N_test = int(N * 0.1)
    N_train = N - N_test

    print('N:{}'.format(N))
    print('N_train:{}'.format(N_train))
    print('N_test:{}'.format(N_test))

    np.random.seed(1701)
    perm = np.random.permutation(N)
    test_indices = perm[:N_test]
    train_indices = perm[N_test:]

    print('train_indices:{}'.format(len(train_indices)))
    print('test_indices:{}'.format(len(test_indices)))

    for i in train_indices:
        datum = json.loads(all_data[i].strip())
        write_line(datum, fp_train)

    for i in test_indices:
        datum = json.loads(all_data[i].strip())
        write_line(datum, fp_test)


def create_kernel(shape, point, radius=kernel_size):
    base = np.zeros(shape)

    x = math.ceil(point[0])
    y = math.ceil(point[1])

    for r in range(shape[0]):
        for c in range(shape[1]):
            base[r, c] = np.exp(-((r-y)**2+(c-x)**2)/radius)

    return base


def load_dataset(is_train=True):
    filename = ""
    if is_train:
        filename = 'dataset/mpii/train_joints.csv'
    else:
        filename = 'dataset/mpii/test_joints.csv'
    fp = open(filename, 'r')
    x = []
    y = []
    while True:
        line = fp.readline()
        line = line.strip('\n')
        if not line:break
        info = line.split(',')
        idx = 0
        if info[0] not in x:
            x.append(info[0])
            y.append([np.reshape(np.asarray([float(data) for data in info[1:]]), (16, 2))])
        else:
            for i, data in enumerate(x):
                if data == info[0]:
                    idx = i
                    break
            y[idx].append(np.reshape(np.asarray([float(data) for data in info[1:]]), (16, 2)))

    return x, y


def parent_point(idx):
    upper_joint = {
        0: 1,
        1: 2,
        2: 6,

        3: 6,
        4: 3,
        5: 4,
        6: 7,
        7: 8,
        8: 9,
        9: 9,
        10: 11,
        11: 12,
        12: 8,
        13: 8,
        14: 13,
        15: 14,
    }
    return upper_joint[idx]


def load_mpii_joints():
    with open('dataset/mpii/joints.json', 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    return json_data


def save_mpii_joints(lines):
    with open('dataset/mpii/joints.json', 'w', encoding='utf-8') as f:
        json.dump(lines, f)


def center_point(points):
    return (np.mean(points[:,0]),np.mean(points[:,1]))


def draw_limb(idx, points, radius):
    src = points[idx]
    # dst = points[parent_point(idx)]
    dst = center_point(points)
    base_map = np.zeros(base_shape)
    if src[0]<dst[0]:
        src_x = math.ceil(src[0])
        dst_x = math.ceil(dst[0])

        src_y = math.ceil(src[1])
        dst_y = math.ceil(dst[1])
    else:
        src_x = math.ceil(dst[0])
        dst_x = math.ceil(src[0])

        src_y = math.ceil(dst[1])
        dst_y = math.ceil(src[1])

    if src_x != dst_x:
        alpha = (dst_y - src_y) / (dst_x - src_x)
        bias = src_y - alpha * src_x
    else:
        alpha = None
        bias = 0

    if alpha is not None:
        for x in range(src_x, dst_x+1):
            if alpha > 0:
                y1 = math.ceil(x*alpha+bias)
                y2 = math.ceil((x+1)*alpha+bias)
                for y in range(y1, y2+1):
                    if (y < src_y or y < dst_y) and (y >= src_y or y >= dst_y):
                        base_map = np.maximum(base_map, create_kernel(base_shape, (x, y), radius))
            elif alpha == 0:
                base_map = np.maximum(base_map, create_kernel(base_shape, (x, src[1]), radius))

            else:
                y1 = math.ceil((x+1) * alpha + bias)
                y2 = math.ceil(x * alpha + bias)
                for y in range(y1,y2+1):
                    if (y < src_y or y < dst_y) and (y >= src_y or y >= dst_y):
                        base_map = np.maximum(base_map, create_kernel(base_shape, (x, y), radius))
    else:
        if src_y<dst_y:
            for y in range(src_y, dst_y+1):
                base_map = np.maximum(base_map, create_kernel(base_shape, (src[0], y), radius))
        else:
            for y in range(dst_y, src_y+1):
                base_map = np.maximum(base_map, create_kernel(base_shape, (src[0], y), radius))

    return base_map


def save_image_total(limbs, index, is_train):
    path = 'train' if is_train else 'validate'
    os.makedirs('dataset/mpii/{0}/center_limb/16'.format(path), exist_ok=True)
    base = np.zeros((64,64))
    limbs = np.asarray(limbs)
    for i in range(0,16):
        base = np.maximum(base, limbs[i,:,:])


    image = decode(base)
    # image = np.asarray(base * 255, dtype=np.uint8)
    image = Image.fromarray(image)
    image.save('dataset/mpii/{1}/center_limb/16/{0}.png'.format(index, path))


def save_images(confidences, limbs, index, is_train):
    path = 'train' if is_train else 'validate'
    for idx in range(len(confidences)):
        heat_dir_path = 'dataset/mpii/{1}/heatmap/{0}'.format(idx,path)
        heat_map_path = 'dataset/mpii/{2}/heatmap/{0}/{1}.png'.format(idx,index,path)
        os.makedirs(heat_dir_path, exist_ok=True)
        # confidence = np.asarray(confidences[idx] * 255, dtype=np.uint8)
        confidence = decode(confidences[idx])
        confidence = np.reshape(confidence, (64, 64))
        image = Image.fromarray(confidence)
        if os.path.isfile(heat_map_path):
            org = Image.open(heat_map_path)
            org = np.asarray(org)
            image = np.maximum(org, image)

        image.save(heat_map_path)


        if idx != 16:
            limb_dir_path = 'dataset/mpii/{1}/center_limb/{0}'.format(idx, path)
            limb_map_path = 'dataset/mpii/{2}/center_limb/{0}/{1}.png'.format(idx,index, path)
            os.makedirs(limb_dir_path, exist_ok=True)
            limb = decode(limbs[idx])
            limb = np.reshape(limb, (64, 64))
            if os.path.isfile(limb_map_path):
                org = Image.open(limb_map_path)
                org = np.asarray(org)
                limb = np.maximum(org, limb)

            image = Image.fromarray(limb)
            image.save(limb_map_path)


def flop_image(confidences,limbs):
    confidences = np.asarray(confidences)
    # print(confidences.shape)
    limbs = np.asarray(limbs)
    # print(limbs.shape)
    c = np.zeros((17,64,64))
    l = np.zeros((16,64,64))
    for idx in range(0,17):
        c[idx,:,:]= cv2.flip(confidences[idx,:,:], 1)
        if idx != 16:
            l[idx,:, :] = cv2.flip(limbs[idx,:, :], 1)
    reverse_c = np.copy(c)
    reverse_l = np.copy(l)
    reverse_c[0] = c[5]
    reverse_c[1] = c[4]
    reverse_c[2] = c[3]

    reverse_c[5] = c[0]
    reverse_c[4] = c[1]
    reverse_c[3] = c[2]

    reverse_c[10] = c[15]
    reverse_c[11] = c[14]
    reverse_c[12] = c[13]

    reverse_c[15] = c[10]
    reverse_c[14] = c[11]
    reverse_c[13] = c[12]

    reverse_l[0] = l[5]
    reverse_l[1] = l[4]
    reverse_l[2] = l[3]

    reverse_l[5] = l[0]
    reverse_l[4] = l[1]
    reverse_l[3] = l[2]

    reverse_l[10] = l[15]
    reverse_l[11] = l[14]
    reverse_l[12] = l[13]

    reverse_l[15] = l[10]
    reverse_l[14] = l[11]
    reverse_l[13] = l[12]
    return reverse_c, reverse_l


def flop_points(points):
    # (0 - r ankle, 1 - r knee, 2 - r hip, 3 - l hip, 4 - l knee, 5 - l ankle, 6 - pelvis, 7 - thorax, 8 - upper neck, 9 - head top, 10 - r wrist, 11 - r elbow, 12 - r shoulder, 13 - l shoulder, 14 - l elbow, 15 - l wrist)
    reverse_points = points
    reverse_points[0] = points[5]
    reverse_points[1] = points[4]
    reverse_points[2] = points[3]

    reverse_points[5] = points[0]
    reverse_points[4] = points[1]
    reverse_points[3] = points[2]

    reverse_points[10] = points[15]
    reverse_points[11] = points[14]
    reverse_points[12] = points[13]
    return reverse_points


def data_generator(batch_size, shuffle=True, is_train=True):
    base_path = root_path+train_path if is_train else root_path+validate_path
    dirs = os.listdir(base_path+'\\image\\')
    # +'0')
    if shuffle:
        random.shuffle(dirs)
    batch_iter = len(dirs)//batch_size
    # heat_map_len = len(os.listdir(base_path + heat_map_path))
    # limb_len = len(os.listdir(base_path + limb_path))
    mean = np.array([0.485, 0.456, 0.406])
    mean = np.array([np.ones((256, 256)) * m for m in mean])
    std = np.array([0.229, 0.224, 0.225])
    std = np.array([np.ones((256, 256)) * s for s in std])
    mean = np.moveaxis(mean, 0, -1)
    std = np.moveaxis(std, 0, -1)
    for idx in range(batch_iter):
        # print(idx)
        x = []
        org_img = []
        heat_maps = []
        limbs = []

        for b_i in range(batch_size):
            file_name = dirs[idx*batch_size + b_i]
            img = Image.open(base_path+'\\image\\'+file_name)
            if img.mode == 'L':
                continue
            # print(img.mode, file_name)
            img = img.resize((256,256))
            org_img.append(img)
            # img.show()
            img = np.asarray(img)/255.
            img = (img - mean)/std
            img = np.moveaxis(img, 2, 0)
            x.append(img)

            if not os.path.exists(base_path+heat_map_path):
                continue
            heat_map_file_path = base_path + heat_map_path + '\\{}\\'.format(0) + file_name
            base_heat = np.asarray(Image.open(heat_map_file_path)) / 255.
            base_heat = np.reshape(base_heat, (64,64,1))

            for h in range(1, heat_map_len):
                heat_map_file_path = base_path+heat_map_path+'\\{}\\'.format(h)+file_name
                heat = np.asarray(Image.open(heat_map_file_path))/255.
                heat = np.reshape(heat, (64,64,1))
                base_heat = np.concatenate([base_heat, heat], axis=-1)

            limb_map_file_path = base_path + limb_path + '\\{}\\'.format(0) + file_name
            base_limb = np.asarray(Image.open(limb_map_file_path)) / 255.
            base_limb = np.reshape(base_limb, (64,64,1))
            for l in range(1, limb_len):
                limb_map_file_path = base_path+limb_path+'\\{}\\'.format(l)+file_name
                limb = np.asarray(Image.open(limb_map_file_path))/255.
                limb = np.reshape(limb, (64,64,1))
                base_limb = np.concatenate([base_limb, limb], axis=-1)

            base_heat = np.moveaxis(base_heat, 2, 0)
            base_limb = np.moveaxis(base_limb, 2, 0)
            heat_maps.append(base_heat)
            limbs.append(base_limb)

        x = np.asarray(x)
        heat_maps = np.asarray(heat_maps)
        limbs = np.asarray(limbs)
        yield x, heat_maps, limbs, org_img


def save_limb(values, path):
    limb_gt = values[:,:,0]
    for k in range(1, 16):
        limb_gt = np.maximum(np.reshape(values[:, :, k], (64,64)), limb_gt)
    limb_gt = decode(limb_gt)
    # limb_gt = np.asarray(((limb_gt+1) * 127.5), dtype=np.uint8)
    limb_gt = np.reshape(limb_gt, (64, 64))
    image = Image.fromarray(limb_gt)
    image.save(path)


def save_heatmap(values, path):
    gt = values[:,:,0]
    for k in range(1, 17):
        gt = np.maximum(np.reshape(values[:, :, k], (64, 64)), gt)
    gt = decode(gt)
    # gt = np.asarray(((gt+1) * 125.), dtype=np.uint8)
    gt = np.reshape(gt, (64, 64))
    image = Image.fromarray(gt)
    image.save(path)
