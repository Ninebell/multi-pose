import os
from conf import *
import numpy as np
from PIL import Image, ImageDraw


def find_point(heat_map):
    point = []
    for r in range(1,63):
        for c in range(1, 63):
            if np.max(heat_map[r-1:r+2,c-1:c+2]) == heat_map[r,c] and heat_map[r,c]>125:
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
            value = np.sum(base * (limb/255)) / calc_dist(g_point, ct_point)

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
        used = [False for _ in range(len(points[idx]))]
        find_max.max_value = 0
        find_max.history = []
        find_max(energy_info, used, 0, 0, [0 for _ in range(len(points[idx]))])

        center_idx.append(find_max.history)
    return center_idx


def result_draw(center, joints, indexing):

    board = np.zeros((64,64,3),dtype=np.uint8)

    img = Image.fromarray(board)
    img_d = ImageDraw.Draw(img)

    center_color = ['red', 'green']

    for i, joint in enumerate(joints):
        for j, point in enumerate(joint):
            center_idx = indexing[i][j]
            img_d.line((point[0], point[1], center[center_idx][0], center[center_idx][1]), width=1, fill=center_color[center_idx])

    img.show()



if __name__ == "__main__":
    test_file = "3.png"
    sample_limb_path = "E:\\dataset\\mpii\\train\\limb_\\"
    sample_heatmap_path = "E:\\dataset\\mpii\\train\\heatmap_\\"
    heat_maps = []
    limbs = []
    for i in range(0,17):
        heat_maps.append(np.array(Image.open(sample_heatmap_path+'{0}\\{1}'.format(i, test_file))))

    for i in range(0,16):
        limbs.append(np.array(Image.open(sample_limb_path+'{0}\\{1}'.format(i, test_file))))

    points = []
    for heat_map in heat_maps:
        points.append(find_point(heat_map))

    for point in points:
        print(point)

    center_idx = make_pair(points[-1], points[:-1], limbs)
    print(center_idx)
    result_draw(points[-1], points[:-1],center_idx)


