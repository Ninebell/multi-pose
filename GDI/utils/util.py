from abc import ABCMeta, abstractmethod
import os


class DataLoader:
    x_label = '/x'
    y_label = '/y'

    def __init__(self, train_path, test_path, same=True):
        DataLoader.path_validate(train_path)
        DataLoader.path_validate(test_path)

        if same:
            DataLoader.check_item_num(train_path)
            DataLoader.check_item_num(test_path)

        self.train_path_x_list = [x_path for x_path in DataLoader.find_path(train_path+DataLoader.x_label)]
        self.train_path_y_list = [y_path for y_path in DataLoader.find_path(train_path+DataLoader.y_label)]

        self.test_path_x_list = [x_path for x_path in DataLoader.find_path(test_path+DataLoader.x_label)]
        self.test_path_y_list = [y_path for y_path in DataLoader.find_path(test_path+DataLoader.y_label)]

    @staticmethod
    def _find_path(path):
        dir_list = os.listdir(path)
        path_list = []
        for file in dir_list:
            full_path = os.path.join(path, file)
            if os.path.isdir(full_path):
                path_list = path_list+DataLoader._find_path(full_path)
            else:
                path_list.append(full_path)
        return path_list

    @staticmethod
    def find_path(path):
        dir_list = os.listdir(path)
        path_list = []
        for file in dir_list:
            full_path = os.path.join(path, file)
            if os.path.isdir(full_path):
                path_list.append(DataLoader._find_path(full_path))

        return path_list

    @staticmethod
    def path_validate(path):
        assert os.path.exists(path+DataLoader.x_label), 'x dir must exist'
        assert os.path.exists(path+DataLoader.y_label), 'y dir must exist'

    @staticmethod
    def check_item_num(path):
        x_pathes = os.listdir(path+DataLoader.x_label)
        count=[]
        for file in x_pathes:
            full_path = os.path.join(path+DataLoader.x_label, file)
            if os.path.isdir(full_path):
                count.append(DataLoader.calculate_item_num(full_path))

        assert len(set(count)) == 1, 'item pairs are not same'

        y_pathes = os.listdir(path+DataLoader.y_label)
        for file in y_pathes:
            full_path = os.path.join(path+DataLoader.y_label, file)
            if os.path.isdir(full_path):
                count.append(DataLoader.calculate_item_num(full_path))

        assert len(set(count)) == 1, 'item pairs are not same'

    @staticmethod
    def calculate_item_num(path):
        dir_list = os.listdir(path)
        count = 0
        for file in dir_list:
            full_path = os.path.join(path, file)
            if os.path.isdir(full_path):
                DataLoader.calculate_item_num(full_path)
            else:
                count = count + 1

        return count
