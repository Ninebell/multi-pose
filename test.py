import os
import cv2
from conf import *
import matplotlib.pyplot as plt

import random

from utils import get_test_set
import torch
import numpy as np
import torch_model.center_net
from PIL import Image, ImageDraw


def find_point(heat_map):
    point = []
    print(np.max(heat_map))
    for r in range(1,63):
        for c in range(1, 63):
            if np.max(heat_map[r-1:r+2,c-1:c+2]) == heat_map[r,c] and heat_map[r,c]>0.5:
                point.append((c,r))
    return point


def calc_dist(pt1, pt2):
    return np.sqrt(np.sum(np.square(np.asarray(pt1) - np.asarray(pt2))))


def calc_energy(center, points, limb):

    energy = []

    for g_point in points:
        pair = []
        for idx, ct_point in enumerate(center):
            base = np.zeros((64,64))
            base = Image.fromarray(base)
            base_draw = ImageDraw.Draw(base)
            base_draw.line((ct_point[0],ct_point[1], g_point[0],g_point[1]), fill=255, width=1)

            # base.show()
            base = np.array(base)/255
            value = np.sum(base * limb) / np.sum(base)

            pair.append((idx, value))
        energy.append(pair)
    return energy


def find_max(energy, used, idx, value, history):
    if idx == len(energy):
        if find_max.max_value < value:
            find_max.max_value = value
            find_max.history = history.copy()

    else:

        for i in range(len(energy[idx])):
            if not used[i]:
                used[i] = True
                history[idx] = i

                find_max(energy, used, idx+1, value+energy[idx][i][1], history)
                used[i] = False


find_max.max_value = 0
find_max.history = 0


def make_pair(center, points, limbs):
    center_idx = []
    for idx in range(len(points)):
        energy_info = calc_energy(center, points[idx], limbs[idx])
        print(idx, energy_info)

        used = [False for _ in range(len(center)+1)]
        find_max.max_value = 0
        find_max.history = []
        find_max(energy_info, used, 0, 0, [-1 for _ in range(len(center)+1)])

        center_idx.append(find_max.history)
    return center_idx


def upper(idx):
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


def result_draw(center, joints, indexing):

    board = np.zeros((256,256,3),dtype=np.uint8)

    img = Image.fromarray(board)
    img_d = ImageDraw.Draw(img)

    center_color = []
    center_group = []
    center_point_group = []
    for i in range(len(center)):
        center_group.append([None for j in range(17)])
        center_color.append((random.randint(0,255), random.randint(0,255), random.randint(0,255)))
        center_point_group.append((random.randint(0,255), random.randint(0,255), random.randint(0,255)))

    for i, joint in enumerate(joints):
        for j, point in enumerate(joint):
            try:
                center_idx = indexing[i][j]
                if center_idx == -1:
                    continue

                center_group[center_idx][i]=[point]
            except:
                continue

    for i in range(len(center)):
        for j in range(len(center_group[i])):
            if center_group[i][j] is not None:
                img_d.ellipse((point[0]*4-5, point[1]*4-5, point[0]*4+5,point[1]*4+5), fill=center_point_group[i])
                # img_d.point((point[0]*4, point[1]*4), fill=center_point_group[i])
                point = center_group[i][j][0]
                upper_idx = upper(j)
                upper_joint = center_group[i][upper_idx]
                if upper_joint is not None:
                    img_d.line((point[0]*4, point[1]*4, upper_joint[0][0]*4, upper_joint[0][1]*4), width=3, fill=center_color[i])

    return np.array(img)


if __name__ == "__main_2_":
    root_path = 'D:\\dataset\\mpii\\train\\input\\'
    heat_path = 'D:\\dataset\\mpii\\train\\heatmap_\\'
    limb_path = 'D:\\dataset\\mpii\\train\\limb_\\'
    file_names = os.listdir(root_path)

    for i in range(len(file_names)):
        img = Image.open('{0}{1}'.format(root_path, file_names[i]))
        input_img = img = img.resize((256, 256))
        points = []

        heat_maps = []
        limbs = []
        for j in range(17):
            heat_maps.append(np.array(Image.open('{0}\\{1}\\{2}'.format(heat_path,j,file_names[i])))/255)

        for j in range(16):
            limbs.append(np.array(Image.open('{0}\\{1}\\{2}'.format(limb_path,j,file_names[i])))/255)

        for heat_map in heat_maps:
            points.append(find_point(heat_map))
        for i in range(0, 17):
            plt.subplot(8, 5, i + 1)
            plt.imshow(heat_maps[i])

        for i in range(0, 16):
            plt.subplot(8, 5, i + 1 + 20)
            plt.imshow(limbs[i])

        center_idx = make_pair(points[-1], points[:-1], limbs)
        print(center_idx)
        result = result_draw(points[-1], points[:-1], center_idx)
        temp = Image.fromarray(result).convert('RGBA')
        input_img = input_img.convert('RGBA')
        blended = Image.blend(input_img, temp, 0.5)
        cv2.imshow("blended", np.array(blended))
        plt.show()

if __name__ == "__main__":
    test_file = "5.png"
    sample_limb_path = "E:\\dataset\\mpii\\test\\limb_\\"
    sample_heatmap_path = "E:\\dataset\\mpii\\test\\heatmap_\\"

    file_names = get_test_set()

    net = torch_model.center_net.CenterNet(256,33, torch.sigmoid)
    net.load_state_dict(torch.load('D:\\dataset\\mpii\\model.dict'))
    # net.load_state_dict('E:\\dataset\\model.dict')

    net = net.cuda()

    for i in range(len(file_names)):
        path = 'E:\\dataset\\mpii\\images\\'
        path = 'E:\\dataset\\mpii\\mpii_human_pose_v1\\images\\'

        img = Image.open('{0}{1}'.format(path, file_names[i]))
        plt.imshow(np.array(img))
        plt.show()
        input_img = img = img.resize((256,256))

        img = np.array(img)/255
        img = np.moveaxis(img, 2, 0)
        print(img.shape)
        img = img.reshape((1,3,256,256))
        input_tensor = torch.from_numpy(img).type(torch.FloatTensor).cuda()
        result = net(input_tensor)

        with torch.no_grad():
            heat_maps = result[1][0,0:17,:,:].cpu().numpy()
            print(heat_maps.shape)
            limbs = result[1][0,17:33,:,:].cpu().numpy()
            for i in range(0, 17):
                plt.subplot(8, 5, i + 1)
                plt.imshow(heat_maps[i])

            for i in range(0, 16):
                plt.subplot(8, 5, i + 1 + 20)
                plt.imshow(limbs[i])
            points = []
            for heat_map in heat_maps:
               points.append(find_point(heat_map))

            for point in points:
                print(point)

            center_idx = make_pair(points[-1], points[:-1], limbs)
            print(center_idx)
            result = result_draw(points[-1], points[:-1],center_idx)
            temp = Image.fromarray(result).convert('RGBA')
            input_img = input_img.convert('RGBA')
            blended = Image.blend(input_img, temp, 0.5)
            cv2_arr = np.array(blended)
            cv2_arr = cv2.cvtColor(cv2_arr, cv2.COLOR_RGB2BGR)
            cv2.imshow("blended", cv2_arr)
            plt.show()
