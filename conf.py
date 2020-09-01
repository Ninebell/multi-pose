import os
from scipy.io import loadmat

data_set_name = 'skeleton'

root_path = 'D:\\dataset\\{0}'.format(data_set_name)
train_path = "\\train\\"
validate_path = "\\validate\\"

heat_map_path = "\\heat\\"
limb_path = "\\limb\\"
#
# train_annotation_path = '\\annotations\\person_keypoints_train2017.json'
# val_annotation_path = '\\annotations\\person_keypoints_val2017.json'


def get_train_data_num():
    base_path = root_path+ train_path + heat_map_path + '0'
    dirs = os.listdir(base_path)
    return len(dirs)


def save_joints():
    joint_data_fn = 'dataset/mpii/data.json'
    # mat = loadmat('E:\\dataset\\mpii\\mpii_human_pose_v1_u12_1.mat')
    mat = loadmat('C:\\Users\\rnwhd\\Desktop\\git\\multi-pose\\dataset\\mpii\\mpii_human_pose_v1_u12_1.mat')
    image_path = 'C:\\Users\\rnwhd\\Desktop\\git\\multi-pose\\dataset\\mpii\\images\\'
    fp = open(joint_data_fn, 'w')

    for i, (anno, train_flag) in enumerate(
            zip(mat['RELEASE']['annolist'][0, 0][0],
                mat['RELEASE']['img_train'][0, 0][0])):

        img_fn = anno['image']['name'][0, 0][0]
        train_flag = int(train_flag)

        head_rect = []
        objpos = anno['annorect']['objpos']
        print(objpos[0][0][0][0][0])
        print(objpos[0][0][0][0][1])

if __name__ =="__main__":
    save_joints()
