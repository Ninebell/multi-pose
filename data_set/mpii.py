import torch
import tqdm
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
                        default=['D:\\dataset\\mpii\\mpii_human_pose_v1_u12_1.mat'], dest='anno', type=str)
    parser.add_argument('--group', '-g', nargs=1, help='dir of mpii group.mat',
                        default=['D:\\dataset\\mpii\\groups_v12.mat'], dest='group', type=str)
    parser.add_argument('--image', '-i', nargs=1, help='dir of mpii input image path',
                        default=['D:\\dataset\\mpii\\images'], dest='img', type=str)
    parser.add_argument('--new', '-n', nargs=1, help='dir of mpii input image path',
                        default=['D:\\dataset\\custom_mpii\\'], dest='new', type=str)

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
    # base = gaussian_filter(base, 1.5)
    # base = base / np.max(base)

    for r in range(shape[0]):
        for c in range(shape[1]):
            base[r, c] = np.exp(-((r-y)**2+(c-x)**2)/5)

    return base


def data_generator(batch_size, data_list, img_path, shuffle=True, is_train=True):
    def open_img(path, mode='r'):
        img = Image.open(path)
        if mode == 'L':
            img = img.convert('L')
        return np.asarray(img)
    mean = np.array([0.485, 0.456, 0.406])
    mean = np.array([np.ones((256,256)) * m for m in mean])
    std = np.array([0.229, 0.224, 0.225])
    std = np.array([np.ones((256, 256)) * s for s in std])
    mean = np.moveaxis(mean, 0, -1)
    std = np.moveaxis(std, 0, -1)
    if shuffle:
        random.shuffle(data_list)

    joint_path = 'train' if is_train else 'validate'

    heat_map_path = img_path + '/{0}/heat/'.format(joint_path)
    limb_map_path = img_path + '/{0}/limb/'.format(joint_path)
    iter_len = len(data_list) // batch_size
    for b in range(iter_len):
        x = []
        org = []
        heat_list = []
        limb_list = []
        for i in range(batch_size):
            heat = []
            limb = []
            b_idx = batch_size*b+i
            data = data_list[b_idx]
            file_name = data['file_name']
            input_img = open_img(img_path + '/image/{0}'.format(file_name), mode='r')
            org.append(input_img.copy())
            if is_train:
                for h in range(17):
                    heat_img = open_img(heat_map_path + '{0}/{1}'.format(h, file_name), mode='L')
                    heat.append(heat_img)

                for l in range(16):
                    limb_img = open_img(limb_map_path+'{0}/{1}'.format(l, file_name), mode='L')
                    limb.append(limb_img)

                heat = np.asarray(heat)
                limb = np.asarray(limb)
                heat_list.append(heat)
                limb_list.append(limb)
            # input_img = (input_img/255 - mean)/std
            input_img = input_img/255
            input_img = np.moveaxis(input_img, 2, 0)
            x.append(input_img)
        x = np.asarray(x)
        heat_list = np.asarray(heat_list)/255
        limb_list = np.asarray(limb_list)/255
        yield x, heat_list, limb_list, org

    return None


def make_limb_map(base, joints):
    center_point = joints[-1]
    for j_idx, joint in enumerate(joints[:-1]):
        new_limb = np.zeros((base.shape[1], base.shape[2]))
        if joint[0] == joint[1] == 0:
            continue

        limb_img = Image.fromarray(new_limb)
        imd = ImageDraw.Draw(limb_img)
        imd.line([joint[0], joint[1], center_point[0], center_point[1]], fill=255, width=1)
        new_limb = np.array(limb_img)
        new_limb = new_limb/np.max(new_limb)

        base[j_idx] = np.maximum(base[j_idx], new_limb)

    return base


def make_new_joints(joints, loc, x_scale, y_scale, limit):
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
        if limit-1 <= math.ceil(joint[0]) <= limit+1:
            joint[0]=limit-1
        if limit-1 <= math.ceil(joint[1]) <= limit+1:
            joint[1]=limit-1
        if math.ceil(joint[0]) > limit or math.ceil(joint[1]) > limit:
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
        kernel = create_kernel((base.shape[1], base.shape[2]), joint)
        base[j_idx] = np.maximum(base[j_idx], kernel)

    return base


def save_label(data_list, save_path, filter_names):
    image_size = 64

    os.makedirs(save_path + '/heat', exist_ok=True)
    os.makedirs(save_path + '/limb', exist_ok=True)
    for i in range(17):
        os.makedirs(save_path+'/heat/{0}'.format(i), exist_ok=True)
    for i in range(16):
        os.makedirs(save_path+'/limb/{0}'.format(i), exist_ok=True)

    iter_len = len(data_list)
    for b in range(iter_len):
        data = data_list[b]

        loc = data['loc']
        persons = data['joints']
        x_scale = image_size / (loc[2] - loc[0])
        y_scale = image_size / (loc[3] - loc[1])

        heat_base = np.zeros((17, image_size, image_size))
        limb_base = np.zeros((16, image_size, image_size))

        for p in persons:
            joints = p['joint_list']
            new_joints = make_new_joints(joints, loc, x_scale, y_scale, image_size)
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
        for i in range(batch_size):
            batch_idx = b * batch_size + i
            data = data_list[batch_idx]
            file_name = data['file_name']
            img = Image.open(image_path+'\\image\\'+file_name)
            base = np.zeros((64,64))
            for i in range(17):
                temp= Image.open(image_path+'train\\heat\\{0}\\'.format(i)+file_name)
                base = np.maximum(base, temp)
            base = image_list_blend(img, base)
            p2 = 'D:\dataset\\custom_mpii\\temp\\'
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

#
# if __name__ == "__main__2":
#     base_path = 'D:\\dataset\\custom_mpii'
#     label_path = base_path
#     labels = os.listdir(base_path+'\\train')
#     train_labels = []
#     validate_labels = []
#
#     filter_names = os.listdir(label_path+'\\trash')
#     for i in range(len(labels)):
#         if not os.path.isfile(os.path.join(label_path+'\\train',labels[i])):
#             continue
#         label = json.load(open(os.path.join(label_path+'\\train',labels[i])))
#         if not label['file_name'] in filter_names:
#             train_labels.append(label)
#
#     for_filter(1, train_labels, 'D:\\dataset\\custom_mpii\\')


def get_datas(anno):
    pred = anno['pred']
    images = pred['image'][0]
    anno_rect = pred['annorect'][0]
    mpii_dict = {}
    for j, (img, rects) in enumerate(zip(images, anno_rect)):
        file_name = img[0, 0][0][0]
        anno_points = rects['annopoints'][0]
        scale = rects['scale'][0]
        obj_pos = rects['objpos'][0]
        anno_rects = {}
        for i, (points, scale, obj) in enumerate(zip(anno_points, scale, obj_pos)):
            rect = {}
            joint_points = {}
            if len(points) == 0:
                continue
            points = points['point']
            point = points[0][0]
            idxes = point['id'][0]
            pt_xs = point['x'][0]
            pt_ys = point['y'][0]
            for index, pt_x, pt_y in zip(idxes, pt_xs, pt_ys):
                joint_points[str(index[0][0])] = [pt_x[0][0], pt_y[0][0]]

            scale = scale[0][0]
            x = obj[0][0][0][0][0]
            y = obj[0][0][1][0][0]
            rect['points'] = joint_points
            rect['scale'] = scale
            rect['obj_pos'] = {'x': x, 'y': y}
            anno_rects[str(i + 1)] = rect

        mpii_dict[str(j)] = {'file_name': file_name, 'anno': anno_rects}
    return mpii_dict


def read_groups(fp):
    ridxes = fp.readline()
    # ridxes = fp.readline()
    ridxes = ridxes.strip().split(',')
    for i in range(len(ridxes)):
        ridxes[i] = ridxes[i].strip('[')
        ridxes[i] = ridxes[i].strip(']')
    return ridxes


def train_data_create(mpii_dict, groups):
    base = 'E:\\dataset\\mpii\\images'
    base = 'E:\\dataset\\mpii\\mpii_human_pose_v1\\images'
    save_base = 'E:\\dataset\\custom_mpii_2\\'

    image_size = 64

    os.makedirs(save_base + '/train/heat', exist_ok=True)
    os.makedirs(save_base + '/train/limb', exist_ok=True)
    for i in range(17):
        os.makedirs(save_base + '/train/heat/{0}'.format(i), exist_ok=True)
    for i in range(16):
        os.makedirs(save_base + '/train/limb/{0}'.format(i), exist_ok=True)

    for i in tqdm.tqdm(range(len(groups))):
        mpii = mpii_dict[str(i)]
        group = groups[i].replace(';',',').split(',')
        file_name = mpii['file_name']
        anno = mpii['anno']
        min_x, max_x, min_y, max_y = 20000, 20000, 0, 0
        mean_scale = 0
        for g in group:
            obj_pos = anno[g]['obj_pos']
            obj_x = obj_pos['x']
            obj_y = obj_pos['y']
            min_x = obj_x if min_x > obj_x else min_x
            min_y = obj_y if min_y > obj_y else min_y

            max_x = obj_x if max_x < obj_x else max_x
            max_y = obj_y if max_y < obj_y else max_y

            mean_scale += anno[g]['scale']

        mean_scale = mean_scale/len(group) if len(group) else 0
        img = Image.open('{0}/{1}'.format(base,file_name))
        # crop_img, loc = ImageCrop(img, [min_x,min_y,max_x,max_y],mean_scale)

        crop_img, loc = ImageCrop(img.copy(), (min_x, min_y, max_x, max_y), mean_scale)
        crop_img = crop_img.resize((256, 256))

        heat_base = np.zeros((17, image_size, image_size))
        limb_base = np.zeros((16, image_size, image_size))

        for g in group:
            joint_loc = anno[g]['points']
            x_scale = image_size / (loc[2] - loc[0])
            y_scale = image_size / (loc[3] - loc[1])
            new_joint = make_new_joints(joint_loc, loc, x_scale, y_scale, image_size)
            heat_base = make_heat_map(heat_base, new_joint)
            limb_base = make_limb_map(limb_base, new_joint)

        for e in range(17):
            base_i = np.array(heat_base[e, :, :]*255, dtype=np.uint8)
            img = Image.fromarray(base_i)
            img.save(save_base+'/train/heat/{0}/{1}'.format(e, file_name))
            for r in range(1, 4):
                r_img = img.rotate(90*r)
                r_img.save(save_base+'/train/heat/{0}/r_{1}_{2}'.format(e, r, file_name))

        for e in range(16):
            base_i = np.array(limb_base[e, :, :]*255, dtype=np.uint8)
            img = Image.fromarray(base_i)
            img.save(save_base+'/train/limb/{0}/{1}'.format(e, file_name))
            for r in range(1, 4):
                r_img = img.rotate(90*r)
                r_img.save(save_base+'/train/limb/{0}/r_{1}_{2}'.format(e, r, file_name))

        crop_img.save(save_base+'/train/image/{0}'.format(file_name))

        for r in range(1, 4):
            r_img = crop_img.rotate(90 * r)
            r_img.save(save_base + '/train/image/r_{1}_{2}'.format(e, r, file_name))


if __name__ == "__main__":
    anno = loadmat('train_anno_list.mat')
    groups = open('train_data_set.csv','r')
    groups = read_groups(groups)
    mpii_dict = get_datas(anno)

    train_data_create(mpii_dict, groups)


if __name__ == "__main__2":
    base_path = 'D:\\dataset\\mpii\\new_data_set'
    label_path = base_path + '\\label'
    labels = os.listdir(label_path+'\\train')
    train_labels = []
    validate_labels = []

    filter_names = os.listdir(label_path+'\\trash')
    for i in range(len(labels)):
        if not os.path.isfile(os.path.join(label_path+'\\train',labels[i])):
            continue
        label = json.load(open(os.path.join(label_path+'\\train',labels[i])))
        if not label['file_name'] in filter_names:
            train_labels.append(label)

    for d, x, y in data_generator(len(train_labels), train_labels, 'D:\\dataset\\custom_mpii\\'):
        d = np.reshape(d, (len(train_labels), 3, -1))
        d = np.moveaxis(d, 0, 1)
        d = np.reshape(d, (3, -1))
        m = np.mean(d, axis=1)
        v = np.var(d, axis=1)
        print(len(train_labels), m, v)


if __name__ == "__main__2":
    print("A")
    mpii_anno_path, mpii_group_path, img_path, new_path = command_parse()
    #
    save_mpii(mpii_anno_path, mpii_group_path)
    # #
    # json_arr = open('custom_json.json', 'r').readline()
    # json_arr = json.loads(json_arr)
    #
    # make_train_data(img_path, new_path, json_arr, (256, 256), 0.2)

    # image_path = new_path+'\\image'
    label_path = new_path

    filter_names = os.listdir(new_path+'\\trash')

    labels = os.listdir(label_path+'\\train')
    train_labels = []
    validate_labels = []
    for i in range(len(labels)):
        if not os.path.isfile(os.path.join(label_path+'\\train',labels[i])):
            continue
        label = json.load(open(os.path.join(label_path+'\\train',labels[i])))
        if label['file_name'] in filter_names:
            continue
        train_labels.append(label)

    batch_size = 1
    save_label(train_labels, new_path+'\\train\\', filter_names)
