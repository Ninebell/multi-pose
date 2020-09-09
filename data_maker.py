from scipy.io import loadmat
import h5py
import os
from PIL import Image, ImageDraw
import math
import numpy as np

import matplotlib.pyplot as plt
import json


def get_joint_from_heatmap(heatmap):
    joints = []
    for r in range(1,63):
        for c in range(1,63):
            if np.max(heatmap[r-1:r+2,c-1:c+2]) == heatmap[r,c] and heatmap[r,c]>0.5:
                joints.append([c,r])
    return joints


def make_pair(joints, centers):
    pairs = []
    for center in centers:
        for joint in joints:
            pairs.append([joint, center])
    return pairs


def calc_energy(pt, center, limb):
    pt = np.asarray(pt)
    center = np.asarray(center)
    gt = draw_limb(pt, center, 64)
    distance = point_distance(pt, center)
    limb_show = limb_to_show(limb)
    # plt.subplot(1,2,1)
    # plt.imshow(limb_show)
    #
    # plt.subplot(1,2,2)
    # plt.imshow(limb_to_show(gt))
    # plt.show()

    energy = np.sum(limb * gt)
    return energy/distance


def find_max_energy(energy, idx, max_idx_list):
    max_idx = np.argmax(energy[idx])
    if energy[idx][max_idx] == 0:
        max_idx_list[idx]=-1
        return
    is_duplicate = np.min(np.abs(max_idx_list - max_idx)) == 0
    if is_duplicate:
        for i, ot_idx in enumerate(max_idx_list):
            if ot_idx == max_idx:
                if energy[i][ot_idx] >= energy[idx][max_idx]:
                    energy[idx][max_idx]=0
                    find_max_energy(energy, idx, max_idx_list)
                else:
                    energy[i][ot_idx]=0
                    max_idx_list[i]=-1
                    find_max_energy(energy, i, max_idx_list)
    else:
        max_idx_list[idx] = max_idx

    return


def find_max_energy_sum(energy):
    max_idx_list = np.zeros((energy.shape[0],), dtype=np.int) - 1
    for idx in range(energy.shape[0]):
        find_max_energy(energy, idx, max_idx_list)
    return max_idx_list


def grouping_joints(joint, center, limb):
    pairs = make_pair(joint, center)
    point_count = len(joint)
    center_count = len(center)

    energy = []
    for pair in pairs:
        energy.append(calc_energy(pair[0], pair[1], limb))

    if len(energy) == 0:
        return np.zeros((center_count,)) - 1

    energy = np.reshape(np.array(energy), (center_count, -1))
    return find_max_energy_sum(energy)


def inference_joints(heatmaps, limbs):
    joints_list = [[] for _ in range(15)]
    for idx, heatmap in enumerate(heatmaps):
        joints = get_joint_from_heatmap(heatmap)
        joints_list[idx] = joints

    max_pair = [[] for _ in range(15)]
    for idx, joints in enumerate(joints_list[:-1]):
        max_pair[idx] = grouping_joints(joints, joints_list[-1], limbs[idx*2:idx*2+2])

    return max_pair, joints_list


def point_distance(pt1, pt2):
    return np.sqrt(np.sum(np.square(pt1 - pt2)))


def squeeze_list(l):
    return np.reshape(np.reshape(np.array(l), (-1,))[0], (-1, ))[0]


class AnnoRect:
    def __init__(self, rect, idx, flag):
        self.scale = 0
        self.objpos = 0
        self.points = {}
        self.head_box = 0
        self.get_anno_rect(rect, idx, flag)

    def get_anno_rect(self, anno_rect, idx, flag):
        self.scale = float(squeeze_list(anno_rect['scale'][idx]))
        self.objpos = (float(squeeze_list(anno_rect['objpos'][idx]['x'])), float(squeeze_list(anno_rect['objpos'][idx]['y'])))

        if not flag:
            return
        self.head_box = ((float(squeeze_list(anno_rect['x1'][idx])), float(squeeze_list(anno_rect['y1'][idx]))),
                         (float(squeeze_list(anno_rect['x2'][idx])), float(squeeze_list(anno_rect['y2'][idx]))))

        anno_points = anno_rect['annopoints']

        point_xs = anno_points[idx]['point'][0][0]['x'][0]
        point_ys = anno_points[idx]['point'][0][0]['y'][0]
        point_id = anno_points[idx]['point'][0][0]['id'][0]

        for point_idx in range(len(point_xs)):
            self.points[int(squeeze_list(point_id[point_idx]))] = (float(squeeze_list(point_xs[point_idx])), float(squeeze_list(point_ys[point_idx])))

    def toJSON(self):
        return {
            'scale': self.scale,
            'objpos': self.objpos,
            'head': self.head_box,
            'points': self.points
        }


class MpiiStruct:
    def __init__(self, name, rects):
        self.name = name
        self.mean_scale = 0
        self.rects = rects

        if len(rects) == 0:
            return

        for rect in self.rects:
            self.mean_scale += rect.scale

        self.mean_scale = self.mean_scale / len(self.rects)

    def toJSON(self):
        return {
            'name': self.name,
            'mean_sacle': self.mean_scale,
            'rect': [rect.toJSON() for rect in self.rects]
        }


def create_multi_rect(info, singles, flag, using_single=False):
    mpii_infos = []
    singles = singles[0]

    name = info['image']['name'][0][0][0]
    if len(info['annorect']) == 0:
        return [MpiiStruct(name, [])]


    rects = info['annorect'][0]

    try:
        scale = rects['scale'][0]
    except:
        # print(rects.dtype)
        print("Except")
        return [MpiiStruct(name, [])]

    if len(rects) == 1:
        mpii_infos.append(MpiiStruct(name, [AnnoRect(rects, 0, flag)]))
        return mpii_infos

    multi_rects = []
    if using_single:
        for single_idx in singles:
            # print(squeeze_list(single_idx))
            if len(single_idx) == 0:
                continue
            single_idx = single_idx[0]
            mpii_infos.append(MpiiStruct(name, [AnnoRect(rects,single_idx-1,flag)]))

    # print(len(rects))

    for idx in range(len(rects)):
        if not using_single:
            try:
                multi_rects.append(AnnoRect(rects, idx, flag))
            except:
                continue

        else:
            if (idx+1) not in singles:
                try:
                    multi_rects.append(AnnoRect(rects, idx, flag))
                except:
                    continue
    if len(multi_rects) != 0:
        mpii_infos.append(MpiiStruct(name, multi_rects))
    return mpii_infos


def pair_joint_to_person(pair, joint, meta, crop_size):
    center_count = len(joint[-1])
    joint_dict = [{} for _ in range (center_count)]

    shape = 64
    crop_size = list(map(int, crop_size.split(',')))
    scale = np.asarray([crop_size[2] - crop_size[0], crop_size[3] - crop_size[1]])/np.array([shape, shape])

    for i in range(15):
        for j, idxes in enumerate(pair[i]):
            if idxes == -1:
                joint_dict[j][i] = None
            else:
                joint_dict[j][i] = np.asarray(joint[i][idxes] * scale, dtype=int) + np.asarray([crop_size[0],crop_size[1]], dtype=int)


def mpii_list_to_hdf5(filename, mpii_list, is_train, data_root):
    i = 0
    with h5py.File(filename.format(data_root), 'w') as hf:
        x_g = hf.create_group('x')
        h_g = hf.create_group('joint')
        c_g = hf.create_group('crop')
        l_g = hf.create_group('limb')
        m_g = hf.create_group('meta')
        for mpii_infos in mpii_list:
            for infos in mpii_infos:
                if not os.path.exists(data_root+'\\images\\{}'.format(infos.name)):
                    print('passed {}'.format(infos.name))
                    continue
                str_json = np.string_(infos.toJSON())

                crop_image, crop_size = image_crop(infos, data_root+'\\images')
                print(crop_size)
                points = filter_not_used(infos, crop_size)

                if len(points) == 0:
                    continue
                points = convert_points(points, crop_size)
                heatmap = points_to_heatmap(points, crop_size)
                limb = points_to_limb(points, crop_size)

                # max_pair, predicted_joints = inference_joints(heatmap, limb)
                # print(len(max_pair), len(max_pair[0]))
                # print(str(crop_size))
                # pair_joint_to_person(max_pair, predicted_joints, str_json, '{},{},{},{}'.format(crop_size[0],crop_size[1],crop_size[2],crop_size[3]))

                print(crop_image.shape)
                x_g.attrs['meta'] = str_json
                # x_g.attrs['crop'] = '{},{},{},{}'.format(crop_size[0],crop_size[1],crop_size[2],crop_size[3])
                c_g.create_dataset(
                    name='C_'+str(i),
                    data=crop_size,
                )
                m_g.create_dataset(
                    name='M_'+str(i),
                    data=str_json
                )

                x_g.create_dataset(
                    name='X_'+str(i),
                    data=crop_image,
                    # shape=(256,256,3),
                    # maxshape=(256,256,3),
                    compression='gzip',
                    compression_opts=9
                )
                if is_train:
                    h_g.create_dataset(
                        name='H_'+str(i),
                        data=heatmap,
                        shape=(15, 64, 64),
                        maxshape=(15, 64,64),
                        compression='gzip',
                        compression_opts=9
                    )
                    l_g.create_dataset(
                        name='L_'+str(i),
                        data=limb,
                        shape=(28, 64, 64),
                        maxshape=(28, 64, 64),
                        compression='gzip',
                        compression_opts=9
                    )

                # if is_train:
                #     heat_blend = image_list_blend(crop_image, heatmap)
                #     limb_show = []
                #     base = np.zeros((64,64))
                #     for j in range(14):
                #
                #         te = limb_to_show(limb[j*2:j*2+2, :, :])
                #         limb_show.append(te)
                #         # plt.imshow(te)
                #         # plt.show()
                #         # plt.close()
                #     limb_blend = image_list_blend(base, limb_show)
                #     plt.subplot(1,2,1)
                #     plt.imshow(heat_blend)
                #     plt.subplot(1,2,2)
                #     plt.imshow(limb_blend)
                #     # plt.savefig(data_root+'\\image_test\\{}.png'.format(i))
                #     plt.show()
                #     plt.close()
                print(i)
                i = i + 1


def get_box(center, size, limit):
    start_x = max(0, center[0] - size//2)
    start_y = max(0, center[1] - size//2)
    end_x = max(limit[1], center[0] - size//2)
    end_y = max(limit[0], center[1] - size//2)
    return start_x, start_y, end_x, end_y


def image_crop(mpii_info, root):
    image_path = '{}\\{}'.format(root, mpii_info.name)
    image = Image.open(image_path)
    image_array = np.asarray(image)

    mean_scale = mpii_info.mean_scale
    rects = mpii_info.rects
    if len(rects) == 0:
        return image_array, (0, 0, 64, 64)

    objpos = np.reshape(np.array(rects[0].objpos), (1,2))
    for rect in rects[1:]:
        objpos = np.concatenate([objpos, np.reshape(rect.objpos, (1,2))], axis=1)
    objpos = objpos.mean(axis=0, dtype=np.int)

    crop_size = int(mean_scale*220)
    start_x, start_y, end_x, end_y = get_box(objpos, crop_size, image_array.shape)
    min_size = min(end_x-start_x, end_y-start_y)
    start_x, start_y, end_x, end_y = get_box(objpos, min_size, image_array.shape)

    crop_image = image_array[start_y:end_y, start_x:end_x, :]
    return crop_image, (start_x, start_y, end_x, end_y)


def filter_not_used(mpii, crop_size):
    rects = mpii.rects
    points_list = []
    for rect in rects:
        points = rect.points
        crop_points = {}
        center_x_1 = math.inf
        center_x_2 = 0
        center_y_1 = math.inf
        center_y_2 = 0
        for pt_key in points:
            pt = list(points[pt_key])
            if pt_key == 6 or pt_key == 7:
                continue
            pt_key = pt_key if pt_key < 6 else pt_key-2

            temp = np.array([int(pt[0]), int(pt[1])])
            if not(crop_size[0] < temp[0] < crop_size[2] and crop_size[1] < temp[1] < crop_size[3]):
                continue

            crop_points[pt_key] = temp
            center_x_1 = min(center_x_1, pt[0])
            center_x_2 = max(center_x_2, pt[0])
            center_y_1 = min(center_y_1, pt[1])
            center_y_2 = max(center_y_2, pt[1])

        pt = np.array([(center_x_1+center_x_2)//2, (center_y_1+center_y_2)//2])
        if center_x_1 != math.inf:
            crop_points[14] = pt
        points_list.append(crop_points)

    return points_list


def convert_points(points_list, crop_size):
    croped_points_list = []
    for points in points_list:
        crop_points = {}
        for pt_key in points:
            pt = list(points[pt_key])
            pt[0] = pt[0] - crop_size[0]
            pt[1] = pt[1] - crop_size[1]
            crop_points[pt_key] = np.array([int(pt[0]), int(pt[1])])

        croped_points_list.append(crop_points)

    return croped_points_list


def create_kernel(shape, point, radius=5):
    base = np.zeros(shape)

    x = int(point[0])
    y = int(point[1])

    for r in range(shape[0]):
        for c in range(shape[1]):
            base[r, c] = np.exp(-((r-y)**2+(c-x)**2)/radius)

    return base


def single_points_from_heatmap(heatmaps):
    pt = []
    for heatmap in heatmaps:
        flatten = np.reshape(heatmap, (-1,))
        idx = np.argmax(flatten)
        y = idx//64
        x = idx%64
        if flatten[idx] > 0.5:
            pt.append(np.array([x,y]))
    return pt


def points_to_heatmap(points_list, crop_size):
    shape = 64
    base = np.zeros((15, shape, shape))

    scale = np.array([shape, shape]) / np.asarray([crop_size[2]-crop_size[0],crop_size[3]-crop_size[1]])
    for points in points_list:
        for pt_key in points:
            # filtering pelvis and thorax
            pt = points[pt_key]

            heat = create_kernel((shape, shape), np.asarray(pt*scale, dtype=int))
            base[pt_key] = np.maximum(base[pt_key], heat)

    return base


def limb_to_show(limb_map):
    x = limb_map[0,:,:]
    y = limb_map[1,:,:]
    divider = np.abs(np.where(x==0,1,x))
    test = np.arctan(np.abs(y)/divider)
    test = test/math.pi

    check_plain_x = np.where(x > 0, 2, -2)
    check_plain_y = np.where(y > 0, 1, -1)
    check_plain = check_plain_x + check_plain_y
    check_plain = np.where(check_plain == 3, 2, check_plain)
    plain = np.where(check_plain == -3, -2, check_plain) + 2
    plain = plain*90
    test = (test*180+plain)/360
    test[0,0]=1

    # plt.imshow(test, cmap='inferno')
    # plt.show()
    return test


def draw_limb(pt, center_pt, shape):
    pt_distance = point_distance(pt, center_pt)
    distance_vector = (pt - center_pt) / pt_distance if pt_distance != 0 else np.asarray([0,0])

    base = np.zeros((shape, shape, 3), dtype=np.uint8)
    img = Image.fromarray(base)
    draw_img = ImageDraw.Draw(img)
    draw_img.line([center_pt[0], center_pt[1], pt[0], pt[1]], fill='white', width=1)
    img = img.convert('L')
    img = np.asarray(img)/255
    img = np.where(img > 0.5, 1, 0)
    base = np.asarray([np.asarray(img), np.asarray(img)])
    base = np.transpose(base, (1,2,0))

    distance_limb = np.transpose(base*distance_vector, (2, 0, 1))
    return distance_limb


def limb_merge(limb_list):
    if len(limb_list) == 0:
        return np.zeros((2,64,64))
    limb_list = np.reshape(np.asarray(limb_list), (-1, 2, 64, 64))
    divider = np.where(limb_list != 0, 1, 0)
    divider = divider.sum(axis=0)
    divider = np.where(divider[0,:,:] == 0, 1, divider[0,:,:])

    # count = np.squeeze(count)
    limb_sum = limb_list.sum(axis=0)
    distance_limb = limb_sum/divider
    return distance_limb


def points_to_limb(points_list, crop_size):
    shape = 64

    temp_limb = [[] for _ in range(14)]
    scale = np.array([shape, shape]) / np.asarray([crop_size[2]-crop_size[0], crop_size[3]-crop_size[1]])
    for points in points_list:
        if len(points) == 0:
            continue
        center_pt = np.asarray(points[14]*scale, dtype=int)
        for pt_key in points:
            # filtering center
            if pt_key == 14:
                continue
            pt = np.asarray(points[pt_key]*scale, dtype=int)
            limb = draw_limb(pt, center_pt, shape)
            temp_limb[pt_key].append(limb)

    limb = limb_merge(temp_limb[0])

    for i in range(1,14):
        temp = limb_merge(temp_limb[i])
        limb = np.concatenate((limb,temp), axis=0)

    return limb


def image_list_blend(org, img_list):
    base = img_list[0]
    for img in img_list:
        base = np.maximum(base, img)
    ary = np.array(base, dtype=np.float)
    plt.imsave('temp.png', ary, cmap='hot')
    temp = Image.open('temp.png')
    temp = temp.resize((64,64))
    temp = temp.convert("RGBA")
    org = Image.fromarray(org)
    org = org.resize((64,64))
    org = org.convert("RGBA")
    blended = Image.blend(org, temp, 0.45)
    return blended


if __name__ == "__main__":
    data_set_root = 'E:\\dataset\\mpii'
    mat = loadmat('{}\\mpii_human_pose_v1_u12_1.mat'.format(data_set_root))
    release = mat['RELEASE']
    anno_list = release['annolist'][0, 0][0]
    img_train = release['img_train'][0, 0][0]
    single_person = release['single_person'][0, 0]


    idx = 0

    test_infos = []
    train_infos = []
    for flag, anno, single in zip(img_train, anno_list, single_person):
        rects = create_multi_rect(anno, single, flag)
        if flag:
            if rects is None:
                continue
            else:
                train_infos.append(rects)
        else:
            # continue
            if rects is None:
                continue
            else:
                test_infos.append(rects)

    print("H")
    os.makedirs('{}\\info'.format(data_set_root), exist_ok=True)
    mpii_list_to_hdf5('{}\\info\\1train_info.h5', train_infos, True, data_set_root)
    mpii_list_to_hdf5('{}\\info\\test_info.h5', test_infos, False, data_set_root)


if __name__ == "__main__1":
    data_set_root = 'E:\\dataset\\mpii'
    train_h5 = h5py.File('{}\\info\\train_info.h5'.format(data_set_root), 'r')
    xs = train_h5['x']
    joints = train_h5['joint']
    limbs = train_h5['limb']
    crop = train_h5['crop']
    test = xs['X_0']
    print(test)
    test = crop['C_7']
    print(test[0], test[1], test[2], test[3])
    print(tuple(test))
