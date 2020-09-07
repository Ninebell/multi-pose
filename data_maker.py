from scipy.io import loadmat
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import math


def squeeze_list(l):
    return np.reshape(np.reshape(np.array(l), (-1,))[0], (-1, ))[0]

def point_distance(pt1, pt2):
    return np.sqrt(np.sum(np.square(pt1 - pt2)))

class AnnoRect:
    def __init__(self, rect, idx, flag):
        self.scale = 0
        self.objpos = 0
        self.points = {}
        self.head_box = 0
        self.get_anno_rect(rect, idx, flag)

    def get_anno_rect(self, anno_rect, idx, flag):
        self.scale = (squeeze_list(anno_rect['scale'][idx]))
        self.objpos = (squeeze_list(anno_rect['objpos'][idx]['x']), squeeze_list(anno_rect['objpos'][idx]['y']))

        if not flag:
            return
        self.head_box = ((squeeze_list(anno_rect['x1'][idx]), squeeze_list(anno_rect['y1'][idx])),
                         (squeeze_list(anno_rect['x2'][idx]), squeeze_list(anno_rect['y2'][idx])))

        anno_points = anno_rect['annopoints']

        point_xs = anno_points[idx]['point'][0][0]['x'][0]
        point_ys = anno_points[idx]['point'][0][0]['y'][0]
        point_id = anno_points[idx]['point'][0][0]['id'][0]

        for point_idx in range(len(point_xs)):
            self.points[squeeze_list(point_id[point_idx])] = (squeeze_list(point_xs[point_idx]), squeeze_list(point_ys[point_idx]))


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


def create_multi_rect(info, singles, flag):
    mpii_infos = []
    singles = singles[0]

    name = info['image']['name'][0][0][0]
    if len(info['annorect']) == 0:
        return MpiiStruct(name, [])

    rects = info['annorect'][0]

    try:
        scale = rects['scale'][0]
    except:
        # print(rects.dtype)
        print("Except")
        return MpiiStruct(name, [])

    multi_rects = []
    for single_idx in singles:
        # print(squeeze_list(single_idx))
        if len(single_idx) == 0:
            continue
        single_idx = single_idx[0]
        mpii_infos.append(MpiiStruct(name, [AnnoRect(rects,single_idx-1,flag)]))

    # print(len(rects))

    for idx in range(len(rects)):
        if (idx+1) not in singles:
            try:
                scale = rects['scale'][idx]
                multi_rects.append(AnnoRect(rects, idx, flag))
            except:
                continue
    if len(multi_rects) != 0:
        mpii_infos.append(MpiiStruct(name, multi_rects))
    return mpii_infos


def image_crop(mpii_info, root):
    image_path = '{}\\{}'.format(root, mpii_info.name)
    image = Image.open(image_path)
    image_array = np.asarray(image)
    print('org', image_array.shape)
    mean_scale = mpii_info.mean_scale
    rects = mpii_info.rects
    objpos = np.reshape(np.array(rects[0].objpos), (1,2))

    for rect in rects[1:]:
        objpos = np.concatenate((objpos, np.reshape(np.array(rect.objpos), (1,2))), axis=0)

    objpos = objpos.mean(axis=0, dtype=np.int)
    crop_size = int(mean_scale*200)
    crop_x = objpos[0] - crop_size//2
    crop_y = objpos[1] - crop_size//2
    start_x = max(0, crop_x)
    start_y = max(0, crop_y)

    end_x = min(image_array.shape[1],objpos[0]+crop_size//2)
    end_y = min(image_array.shape[0],objpos[1]+crop_size//2)
    min_size = min(end_y - start_y, end_x - start_x)

    crop_image = image_array[start_y:start_y+min_size//2, start_x:start_x+min_size//2, :]
    print(crop_image.shape)

    return crop_image, (start_x, start_y, min_size)


def convert_points(mpii, crop_size):
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
            pt[0] = pt[0] - crop_size[0]
            pt[1] = pt[1] - crop_size[1]
            crop_points[pt_key] = np.array([pt[0], pt[1]])

            center_x_1 = min(center_x_1, pt[0])
            center_x_2 = max(center_x_2, pt[0])
            center_y_1 = min(center_y_1, pt[1])
            center_y_2 = max(center_y_2, pt[1])

        pt = np.array([(center_x_1+center_x_2)//2, (center_y_1+center_y_2)//2])
        crop_points[14] = pt
        points_list.append(crop_points)

    return points_list


def create_kernel(shape, point, radius=5):
    base = np.zeros(shape)

    x = point[0]
    y = point[1]

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

    scale = np.array([shape, shape]) / crop_size[2]
    for points in points_list:
        for pt_key in points:
            # filtering pelvis and thorax
            pt = points[pt_key]

            heat = create_kernel((shape, shape), np.asarray(pt*scale, dtype=float))
            base[pt_key] = np.maximum(base[pt_key], heat)

    return base


def limb_to_show(limb_map):
    x = limb_map[0,:,:]
    y = limb_map[1,:,:]
    divider = np.abs(np.where(x==0,1,x))
    test = np.arctan(np.abs(y)/divider)
    test = test/math.pi

    check_plain_x = np.where(x>0, 2, -2)
    check_plain_y = np.where(y>0, 1, -1)
    check_plain = check_plain_x + check_plain_y
    check_plain = np.where(check_plain == 3, 2, check_plain)
    plain = np.where(check_plain == -3, -2, check_plain) + 2
    plain = plain*90
    test = (test*180+plain)/360
    test[0,0]=1
    plt.imshow(test, cmap='inferno')
    plt.show()
    print(np.max(test), np.min(test))
    print(test.shape)


def draw_limb(pt, center_pt, shape):
    pt_distance = point_distance(pt, center_pt)
    distance_vector = (pt - center_pt) / pt_distance

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
    count = np.where(divider[0,:,:] == 0, 1, divider[0,:,:])

    count = np.squeeze(count)
    limb_sum = limb_list.sum(axis=0)
    distance_limb = limb_sum/count
    return distance_limb


def points_to_limb(points_list, crop_size):
    shape = 64

    temp_limb = [[] for _ in range(14)]
    scale = np.array([shape, shape]) / crop_size[2]
    for points in points_list:
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
        print(temp.shape)
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
    plt.imshow(blended)
    plt.show()


if __name__ == "__main__":
    mat = loadmat('E:\\dataset\\mpii\\mpii_human_pose_v1_u12_1.mat')
    release = mat['RELEASE']
    anno_list = release['annolist'][0, 0][0]
    img_train = release['img_train'][0, 0][0]
    single_person = release['single_person'][0, 0]

    print(point_distance(np.array([0,0]), np.array([2,2])))

    idx = 0

    data_root = 'E:\\dataset\\mpii\\mpii_human_pose_v1\\images'

    for flag, anno, single in zip(img_train, anno_list, single_person):
        if flag:
            mpii_infos = create_multi_rect(anno, single, flag)
            for infos in mpii_infos:

                crop_image, image_size = image_crop(infos, data_root)
                print(crop_image.shape, image_size)

                points = convert_points(infos, image_size)
                heatmap = points_to_heatmap(points, image_size)
                limb = points_to_limb(points, image_size)
                print('shape', heatmap.shape, limb.shape)

                image_list_blend(crop_image, heatmap)

                points = single_points_from_heatmap(heatmap)
                print(points)
                image_path = '{}\\{}'.format(data_root, infos.name)
                image = Image.open(image_path)
                image_array = np.asarray(image)
                image_draw = ImageDraw.Draw(image)

                for point in points[:-1]:
                    point = point*(image_size[2]/64)
                    point = point+image_size[:2]
                    image_draw.ellipse((point[0], point[1],point[0]+30, point[1]+30), fill='red')

                point = points[-1]
                point = point * (image_size[2] / 64)
                point = point + image_size[:2]
                image_draw.ellipse((point[0], point[1],point[0]+30, point[1]+30), fill='green')

                point = infos.rects[0].objpos
                image_draw.ellipse((point[0], point[1],point[0]+30, point[1]+30), fill='blue')

                image_array = np.asarray(image)
                plt.imshow(image_array)
                plt.show()
                plt.close()


                # heatmae = points_to_heatmap(infos.rects[0].points, image_array.shape, True)
                # image_list_blend(image_array, heatmap)

