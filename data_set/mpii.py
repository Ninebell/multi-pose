import torch
import os
import shutil
import json
import argparse
import numpy as np
import math
import cv2

import matplotlib.pyplot as plt


from PIL import Image, ImageDraw
import random
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter

joint_id = {'r_ankle': 0, 'r_knee': 1, 'r_hip': 2, 'l_hip': 3, 'l_knee': 4, 'l_ankle': 5,
            'pelvis': 6, 'thorax': 7, 'upper_neck': 8, 'heat_top': 9,
            'r_wrist': 10, 'r_elbow': 11, 'r_shoulder': 12, 'l_shoulder': 13, 'l_elbow': 14, 'l_wrist': 15}

mpii_eval_joint = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13, 14, 15]

custom_json_sk = {'image_id': '', 'anno_rects': [], 'groups': []}
anno_rect_sk = {'joint_list': [], 'obj_pos': [], 'visible': [], 'scale': []}

train_sk = {
    'file_name': '',
    'mean_scale': 0,
    'loc': 0,
    'org_size': [],
    'joints': [],
    'is_train': 0
}


def command_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--anno', '-a', nargs=1, help='dir of mpii anno.mat',
                        default=['E:\\dataset\\mpii\\mpii_human_pose_v1_u12_1.mat'], dest='anno', type=str)
    parser.add_argument('--group', '-g', nargs=1, help='dir of mpii group.mat',
                        default=['E:\\dataset\\mpii\\groups_v12.mat'], dest='group', type=str)
    parser.add_argument('--image', '-i', nargs=1, help='dir of mpii input image path',
                        default=['E:\\dataset\\mpii\\images'], dest='img', type=str)
    parser.add_argument('--new', '-n', nargs=1, help='dir of mpii input image path',
                        default=['E:\\dataset\\mpii\\new_dataset'], dest='new', type=str)

    args = parser.parse_args()

    mpii_anno = args.anno[0]
    mpii_group = args.group[0]
    mpii_img = args.img[0]
    new_path = args.new[0]

    return mpii_anno, mpii_group, mpii_img, new_path


def get_visible(j_id, annopoint):
    if 'is_visible' in str(annopoint.dtype):
        vis = [v[0] if v else [0]
               for v in annopoint['is_visible'][0]]
        vis = dict([(k, int(v[0])) if len(v) > 0 else v
                    for k, v in zip(j_id, vis)])
    else:
        vis = None
    return vis


def get_joint_pos_id(annopoint):
    if len(annopoint) != 0:
        j_id = [str(j_i[0, 0]) for j_i in annopoint['id'][0]]
        x = [x[0, 0] for x in annopoint['x'][0]]
        y = [y[0, 0] for y in annopoint['y'][0]]

    joint_pos = {}
    for _j_id, (_x, _y) in zip(j_id, zip(x, y)):
        joint_pos[str(_j_id)] = [float(_x), float(_y)]
    return j_id, joint_pos


def save_mpii(anno_path, group_path):
    anno = loadmat(anno_path)
    groups = loadmat(group_path)
    json_arr = []

    for i, (anno, train_flag, groups) in enumerate(
            zip(anno['RELEASE']['annolist'][0, 0][0],
                anno['RELEASE']['img_train'][0, 0][0],
                groups['groups'])):
        train_flag = int(train_flag)

        if not train_flag:
            continue

        img_fn = anno['image']['name'][0, 0][0]
        custom_json = custom_json_sk.copy()
        json_arr.append(custom_json)
        inner_group = []
        if len(groups[0]) != 0:
            for group in groups[0][0]:
                squeezed = np.squeeze(group).tolist()
                inner_group.append(squeezed if type(squeezed) != int else [squeezed])

        custom_json['groups'] = inner_group

        custom_json['image_id'] = img_fn

        if 'annopoints' in str(anno['annorect'].dtype):
            annopoints = anno['annorect']['annopoints'][0]
            scales = anno['annorect']['scale'][0]
            objposes = anno['annorect']['objpos'][0]

            rects = {}

            j = 0
            for annopoint, scale, objpos in zip(
                    annopoints, scales, objposes):

                if len(annopoint) != 0:
                    anno_rect = anno_rect_sk.copy()
                    annopoint = annopoint['point'][0, 0]
                    j_id, joint_pos = get_joint_pos_id(annopoint)
                    anno_rect['joint_list'] = joint_pos

                    vis = get_visible(j_id, annopoint)

                    anno_rect['visible'] = vis
                    anno_rect['scale'] = float(scale[0][0])
                    anno_rect['obj_pos'] = (int(objpos[0][0][0][0][0]), int(objpos[0][0][1][0][0]))
                    rects[str(j)] = anno_rect
                    j = j + 1
            custom_json['anno_rects'] = rects

    with open('custom_json.json', 'w') as json_file:
        json.dump(json_arr, json_file)

    return json_arr


def ImageCrop(img, loc, scale):
    scale_v = 100 * scale
    min_x = loc[0] - scale_v if loc[0] - scale_v >= 0 else 0
    min_y = loc[1] - scale_v if loc[1] - scale_v >= 0 else 0

    max_x = loc[2] + scale_v if loc[2] + scale_v < img.size[0] else img.size[0]
    max_y = loc[3] + scale_v if loc[3] + scale_v < img.size[1] else img.size[1]

    return img.crop((min_x, min_y, max_x, max_y)), (min_x, min_y, max_x, max_y)


def make_train_data(root_path, new_path, json_arr, size, ratio):
    print(len(json_arr))

    train_data = []
    validate_data = []

    train_num = 0
    validate_num = 0
    file_idx = 0
    for js in json_arr:
        file_name = js['image_id']
        file_path = os.path.join(root_path, file_name)
        try:
            img = Image.open(file_path)
        except:
            continue

        rect = js['anno_rects']
        if len(rect) == 0:
            continue
        else:
            print(js['groups'])
            print(rect.keys())

            for groups in js['groups']:
                if len(groups) == 0:
                    continue
                j = train_sk.copy()
                is_train = random.random() > ratio
                j['is_train'] = is_train

                if is_train:
                    train_num = train_num + 1
                else:
                    validate_num = validate_num + 1

                j['file_name'] = '{1}_{0}'.format(file_name, len(groups))
                file_idx = file_idx + 1
                mean_scale = 0
                min_x, min_y, max_x, max_y = 20000, 20000, 0, 0
                error = False
                for idx in groups:
                    idx = str(idx - 1)
                    if idx not in rect.keys():
                        error = True
                        continue
                    j['joints'].append(rect[idx])
                    obj_x = rect[idx]['obj_pos'][0]
                    obj_y = rect[idx]['obj_pos'][1]
                    min_x = obj_x if min_x > obj_x else min_x
                    min_y = obj_y if min_y > obj_y else min_y

                    max_x = obj_x if max_x < obj_x else max_x
                    max_y = obj_y if max_y < obj_y else max_y
                    scale = rect[idx]['scale']
                    mean_scale += scale if scale >= 2 else 2.
                if error:
                    continue
                mean_scale /= len(groups)
                # center = ((max_x+min_x)/2, (max_y+min_y)/2)
                crop_img, loc = ImageCrop(img.copy(), (min_x, min_y, max_x, max_y), mean_scale)
                crop_img = crop_img.resize(size)
                j['mean_scale'] = mean_scale
                j['org_size'] = img.size
                j['loc'] = loc

                crop_img.save('{0}/image/{1}'.format(new_path, j['file_name']))
                mid = 'train' if is_train else 'validate'
                with open('{0}/label/{1}/{2}.json'.format(new_path, mid, j['file_name'].split('.')[0]), 'w') as json_file:
                    json.dump(j, json_file)
                j['joints'].clear()

    print(train_num, validate_num)


def create_kernel(shape, point):

    base = np.zeros(shape)

    x = math.ceil(point[0])
    y = math.ceil(point[1])
    base[y,x]=1
    base = gaussian_filter(base, 3)
    base = base / np.max(base)

    # for r in range(shape[0]):
    #     for c in range(shape[1]):
    #         base[r, c] = np.exp(-((r-y)**2+(c-x)**2)/9)

    return base


def data_generator(batch_size, data_list, image_path):
    random.shuffle(data_list)
    iter_len = len(data_list) // batch_size
    for b in range(iter_len):
        x = []
        y = []
        for i in range(batch_size):
            batch_idx = b*batch_size+i
            data = data_list[batch_idx]
            file_name = data['file_name']
            img = Image.open(os.path.join(image_path, file_name))
            mean_scale = data['mean_scale']
            loc = data['loc']
            persons = data['joints']
            x_scale = 256/(loc[2] - loc[0])
            y_scale = 256/(loc[3] - loc[1])

            base = np.zeros((256,256))

            print(loc, x_scale, y_scale)
            for p in persons:
                joints = p['joint_list']
                for joint_idx in joints.keys():
                    joint = joints[joint_idx]
                    visible = p['visible'][joint_idx]
                    print(joint)
                    joint = [joint[0]-loc[0], joint[1]-loc[1]]
                    joint = [joint[0]*x_scale, joint[1]*y_scale]
                    if math.ceil(joint[0]) > 255.5 or math.ceil(joint[1]) > 255.5:
                        continue
                    kernel = create_kernel((256, 256), joint)
                    base = np.maximum(base, kernel)
            base = image_list_blend(img, base)
            base = base.convert("RGB")
            cvm = np.asarray(base) * 255
            cvm = np.asarray(cvm, dtype=np.uint8)
            cvm = cv2.cvtColor(cvm, cv2.COLOR_RGB2BGR)
            cv2.imshow("tuple", cvm)
            key = cv2.waitKey()
            if chr(key) == 'b':
                p1 = 'E:\dataset\\mpii\\new_dataset\\label\\train\\'
                p2 = 'E:\dataset\\mpii\\new_dataset\\label\\trash\\'
                fn = file_name.split('.')[0]+'.json'
                shutil.move(p1+fn, p2+fn)

                base.save(p2+file_name)

            # fig = plt.figure()
            # ax1 = fig.add_subplot(2,1,1)
            # ax2 = fig.add_subplot(2,1,2)
            #
            # ax1.imshow(img)
            # ax2.imshow(base)
            # plt.show()
            # plt.close()


def make_limb_map(base, joints):
    center_point = joints[-1]
    for j_idx, joint in enumerate(joints[:-1]):
        new_limb = np.zeros((256,256))
        if joint[0] == joint[1] == 0:
            continue

        limb_img = Image.fromarray(new_limb)
        imd = ImageDraw.Draw(limb_img)
        imd.line([joint[0], joint[1], center_point[0], center_point[1]], fill=255, width=2)
        new_limb = np.array(limb_img)
        new_limb = new_limb/np.max(new_limb)

        base[j_idx] = np.maximum(base[j_idx], new_limb)

    return base


def make_new_joints(joints, loc, x_scale, y_scale):
    new_joints = np.zeros((17,2))
    center_x = 0
    center_y = 0
    using_joints = 0
    for joint_idx in joints.keys():
        j_idx = int(joint_idx)
        joint = joints[joint_idx]
        joint = [joint[0] - loc[0], joint[1] - loc[1]]
        joint = [joint[0] * x_scale, joint[1] * y_scale]
        if -2 < joint[0] < 0:
            joint[0] = 0
        if -2 < joint[1] < 0:
            joint[1] = 0

        if math.ceil(joint[0]) > 255.0 or math.ceil(joint[1]) > 255.0:
            continue
        if joint[0] < 0 or joint[1] < 0:
            continue

        center_x = center_x + joint[0]
        center_y = center_y + joint[1]
        using_joints = using_joints+1

        new_joints[j_idx, 0] = joint[0]
        new_joints[j_idx, 1] = joint[1]
        # kernel = create_kernel((256, 256), joint)
        # base[j_idx] = np.maximum(base[j_idx], kernel)

    new_joints[16, 0] = center_x/using_joints if using_joints != 0 else 0
    new_joints[16, 1] = center_y/using_joints if using_joints != 0 else 0
    return new_joints


def make_heat_map(base, joints):
    for j_idx, joint in enumerate(joints):
        if joint[0] == joint[1] == 0:
            continue
        kernel = create_kernel((256, 256), joint)
        base[j_idx] = np.maximum(base[j_idx], kernel)

    return base


def save_label(data_list, save_path, filter_names):
    os.makedirs(save_path + '/heat', exist_ok=True)
    os.makedirs(save_path + '/limb', exist_ok=True)
    for i in range(17):
        os.makedirs(save_path+'/heat/{0}'.format(i), exist_ok=True)
    for i in range(16):
        os.makedirs(save_path+'/limb/{0}'.format(i), exist_ok=True)

    iter_len = len(data_list) // batch_size
    for b in range(iter_len):
        for i in range(batch_size):

            batch_idx = b * batch_size + i
            data = data_list[batch_idx]

            if data['file_name'] in filter_names:
                continue

            loc = data['loc']
            persons = data['joints']
            x_scale = 256 / (loc[2] - loc[0])
            y_scale = 256 / (loc[3] - loc[1])

            heat_base = np.zeros((17, 256, 256))
            limb_base = np.zeros((16, 256, 256))

            for p in persons:
                joints = p['joint_list']
                new_joints = make_new_joints(joints, loc, x_scale, y_scale)
                heat_base = make_heat_map(heat_base, new_joints)
                limb_base = make_limb_map(limb_base, new_joints)

            for e in range(17):
                base_i = np.array(heat_base[e, :, :]*255, dtype=np.uint8)
                img = Image.fromarray(base_i)
                img.save(save_path+'/heat/{0}/{1}'.format(e, data['file_name']))

            for e in range(16):
                base_i = np.array(limb_base[e, :, :]*255, dtype=np.uint8)
                img = Image.fromarray(base_i)
                img.save(save_path+'/limb/{0}/{1}'.format(e, data['file_name']))


def for_filter(batch_size, data_list, image_path):
    iter_len = len(data_list) // batch_size
    for b in range(iter_len):
        x = []
        y = []
        for i in range(batch_size):
            print(b)
            batch_idx = b * batch_size + i
            data = data_list[batch_idx]
            file_name = data['file_name']
            img = Image.open(os.path.join(image_path, file_name))
            loc = data['loc']
            persons = data['joints']
            x_scale = 256 / (loc[2] - loc[0])
            y_scale = 256 / (loc[3] - loc[1])

            base = np.zeros((256, 256))

            for p in persons:
                joints = p['joint_list']
                for joint_idx in joints.keys():
                    joint = joints[joint_idx]
                    joint = [joint[0] - loc[0], joint[1] - loc[1]]
                    joint = [joint[0] * x_scale, joint[1] * y_scale]
                    if -2<joint[0]<0:
                        joint[0]=0
                    if -2<joint[1]<0:
                        joint[1]=0
                    if math.ceil(joint[0]) > 255.5 or math.ceil(joint[1]) > 255.5:
                        continue
                    if joint[0]<0 or joint[1]<0:
                        continue
                    kernel = create_kernel((256, 256), joint)
                    base = np.maximum(base, kernel)
            base = image_list_blend(img, base)
            p2 = 'E:\dataset\\mpii\\new_dataset\\label\\temp\\'
            base = base.convert('RGB')
            base.save(p2+file_name)


def get_joints_from_heat_map(heat_maps, limb_maps):
    max_pool = torch.nn.MaxPool2d((3,3),stride=1,padding=1)
    max_values = max_pool(heat_maps)
    batch_size = max_values.shape[0]
    ones = torch.ones(max_values.shape)
    zeros = torch.zeros(max_values.shape)
    expect_points = torch.where(heat_maps == max_values)
    # expect_points = expect_points.view([1, 17, -1])
    indexes = torch.argmax(expect_points, dim=2)
    print(indexes.shape)
    print(indexes.numpy())


def point_rescale(points, data):
    rescaled_points = []
    loc = data['loc']
    start_x = data['loc'][0]
    start_y = data['loc'][1]
    x_scale = (loc[2] - loc[0]) / 256
    y_scale = (loc[3] - loc[1]) / 256
    for point in points:
        new_point = (point[0] * x_scale + start_x, point[1] * y_scale + start_y)
        rescaled_points.append(new_point)
    return rescaled_points


def image_list_blend(org, base):
    ary = np.array(base, dtype=np.float)
    # ary /= 255.
    plt.imsave('temp.png',ary, cmap='hot')
    temp = Image.open('temp.png')
    temp = temp.resize((256,256))
    temp = temp.convert("RGBA")
    t = org.convert("RGBA")
    blended = Image.blend(t, temp, 0.5)
    return blended


if __name__ == "__main__":
    mpii_anno_path, mpii_group_path, img_path, new_path = command_parse()
    #
    # save_mpii(mpii_anno_path, mpii_group_path)
    # #
    # json_arr = open('custom_json.json', 'r').readline()
    # json_arr = json.loads(json_arr)
    #
    # make_train_data(img_path, new_path, json_arr, (256, 256), 0.2)

    image_path = new_path+'\\image'
    label_path = new_path+'\\label'

    filter_names = os.listdir(label_path+'\\trash')

    labels = os.listdir(label_path+'\\train')
    train_labels = []
    validate_labels = []
    for i in range(len(labels)):
        if not os.path.isfile(os.path.join(label_path+'\\train',labels[i])):
            continue
        label = json.load(open(os.path.join(label_path+'\\train',labels[i])))
        train_labels.append(label)

    batch_size = 1
    save_label(train_labels, label_path+'\\train', filter_names)
