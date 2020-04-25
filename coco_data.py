from conf import *
import os
import json
from PIL import Image, ImageDraw
from scipy.ndimage.filters import gaussian_filter
import numpy as np
from tqdm import tqdm


def load_annotaion():
    json_data = open(root_path+train_annotation_path, 'r').read()
    data = json.loads(json_data)
    return data['annotations']


def create_kernel(shape, key_points):
    base = np.zeros(shape)
    return base


def heat_map_create(annotation_dict, shape):
    file_name_format = "{0:012d}.jpg"

    def __get_image(id):
        img_file_path = root_path+train_path+file_name_format.format(id)
        img = Image.open(img_file_path)
        return np.array(img)

    def __change_to_heat_map(shape, key_points, box, org_shape):
        key = np.asarray(key_points).reshape((-1, 3))
        base = np.zeros((org_shape[0], org_shape[1], key.shape[0]))
        sgm = np.log(box[2] + box[3])/2
        for i in range(key.shape[0]):
            x = int(key[i,0])
            y = int(key[i,1])

            if x == y == 0:
                continue
            base[y,x,i] = 1
            base[:,:,i] = gaussian_filter(base[:,:,i],sigma=sgm)
            base[:,:,i] = base[:,:,i]/np.max(base[:,:,i])
        return base

    os.makedirs(root_path+label_path,exist_ok=True)

    for i in tqdm(range(len(annotation_dict))):
        img_id = annotation_dict[i]['image_id']
        box = annotation_dict[i]['bbox']
        key_points = annotation_dict[i]['keypoints']
        num_key_points = annotation_dict[i]['num_keypoints']
        if num_key_points < 10:
            continue

        img = __get_image(img_id)
        heat_maps = __change_to_heat_map(shape, key_points, box, img.shape)

        for j in range(heat_maps.shape[2]):

            j_label_path = root_path + label_path+"\\{}\\".format(j)
            img_file_path = j_label_path+file_name_format.format(img_id)
            os.makedirs(j_label_path, exist_ok=True)

            heat_map = np.asarray(heat_maps[:, :, j] * 255, dtype=np.uint8)

            if os.path.exists(img_file_path):
                org = np.asarray(Image.open(img_file_path))
                heat_map = np.maximum(org,heat_map)

            Image.fromarray(heat_map).save(img_file_path)


if __name__ == "__main__":
    anno = load_annotaion()
    heat_map_create(anno, (64,64))
