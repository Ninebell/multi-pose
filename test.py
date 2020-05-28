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


def calc_maximum_energy(center, points, limb):

    energy = []

    Image.fromarray(limb).show()
    for idx, ct_point in enumerate(center):
        for g_point in points:
            base = np.zeros((64,64))
            base = Image.fromarray(base)
            base_draw = ImageDraw.Draw(base)
            base_draw.line((ct_point[0],ct_point[1], g_point[0],g_point[1]), fill=255, width=1)

            # base.show()
            base = np.array(base)/255
            Image.fromarray((base*limb/255)*255).show()
            value = np.sum(base * (limb/255)) / calc_dist(g_point, ct_point)

            energy.append((idx, value))


    print(energy)


def make_pair(center, points, limbs):
    for idx in range(len(points)):
        calc_maximum_energy(center, points[idx], limbs[idx])



if __name__ == "__main__":
    test_file = "3.png"
    sample_limb_path = "D:\\dataset\\mpii\\train\\limb_\\"
    sample_heatmap_path = "D:\\dataset\\mpii\\train\\heatmap_\\"
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

    make_pair(points[-1], points[:1], limbs)


