import os
import json
import argparse
import numpy as np
from PIL import Image
import random
from scipy.io import loadmat

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
                        default=['D:\\dataset\\mpii\\t_images'], dest='new', type=str)

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
                with open('{0}/label/{1}.json'.format(new_path, j['file_name'].split('.')[0]), 'w') as json_file:
                    json.dump(j, json_file)
                j['joints'].clear()

    print(train_num, validate_num)


def data_generator(batch_size, train_list):
    random.shuffle(train_list)


if __name__ == "__main__":
    mpii_anno_path, mpii_group_path, img_path, new_path = command_parse()

    json_arr = open('custom_json.json', 'r').readline()
    json_arr = json.loads(json_arr)

    make_train_data(img_path, new_path, json_arr, (256, 256), 0.2)
