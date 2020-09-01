from scipy.io import loadmat
import numpy as np


def squeeze_list(l):
    return np.reshape(np.reshape(np.array(l), (-1,))[0], (-1, ))[0]


class AnnoRect:
    def __init__(self, rect, idx, flag):
        self.get_anno_rect(rect, idx, flag)
        self.scale = 0
        self.objpos = 0
        self.points = {}
        self.head_box = 0

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


if __name__ == "__main__":
    mat = loadmat('E:\\dataset\\mpii\\mpii_human_pose_v1_u12_1.mat')
    release = mat['RELEASE']
    anno_list = release['annolist'][0, 0][0]
    img_train = release['img_train'][0, 0][0]
    single_person = release['single_person'][0, 0]

    idx = 0

    for flag, anno, single in zip(img_train, anno_list, single_person):
        mpii_infos = create_multi_rect(anno, single, flag)

