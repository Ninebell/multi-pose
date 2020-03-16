from abc import ABCMeta, abstractmethod
import math
import numpy as np


class BaseGenerator(metaclass=ABCMeta):
    def __init__(self, x, y, length, batch_size=1, shuffle=False, validate_ratio=0.0, name='Generator'):
        assert validate_ratio < 1, 'validate ratio can not more 1.'
        self.x = x
        self.y = y
        self.shuffle = shuffle
        self.batch_size = batch_size
        indexes = np.arange(0, length)
        np.random.shuffle(indexes)
        self.train_index = 0
        self.validate_index = 0
        self.name = name
        validate_line = math.ceil(length*(1-validate_ratio))
        self.train_indexes = indexes[:validate_line]
        self.validate_indexes = indexes[validate_line:]
        self.__train_len = len(self.train_indexes)
        self.__validate_len = len(self.validate_indexes)

        self.validate_ratio = validate_ratio
        self.print_info()

    def train_length(self):
        return len(self.train_indexes)//self.batch_size

    def validate_length(self):
        return len(self.validate_indexes)//self.batch_size

    def print_info(self):
        x_shape_str = "{0}".format(self.x.shape)
        y_shape_str = "{0}".format(self.y.shape)
        validate_str = "{0:1.2f}".format(self.validate_ratio)
        line = "#"*5
        title = "{0}{1:^41s}{2}".format(line,self.name,line)
        line = "#"*51
        print(title)
        print("#{0:^24s}:{1:^24d}#".format("BatchSize", self.batch_size))
        print("#{0:^24s}:{1:^24s}#".format("X Shape", x_shape_str))
        print("#{0:^24s}:{1:^24s}#".format("Y Shape", y_shape_str))
        print("#{0:^24s}:{1:^24s}#".format("Validation", validate_str ))
        print(line)

    def iteration_end(self):
        if self.shuffle:
            np.random.shuffle(self.train_indexes)
        self.train_index = 0

    def validate_iteration_end(self):
        if self.shuffle:
            np.random.shuffle(self.validate_indexes)
        self.validate_index = 0

    def update_iteration(self):
        self.train_index = (self.train_index + self.batch_size) % self.__train_len

    def update_validate_iteration(self):
        self.validate_index = (self.validate_index + self.batch_size) % self.__validate_len

    def generator(self):
        if self.train_index + self.batch_size > self.__train_len:
            self.iteration_end()
        x, y = self.__generator__()
        self.update_iteration()
        return x, y

    def validate_generator(self):
        if self.train_index + self.batch_size > self.__validate_len:
            self.iteration_end()
        x, y = self.__validate_generator__()
        self.update_validate_iteration()
        return x, y

    @abstractmethod
    def __generator__(self):
        ...

    @abstractmethod
    def __validate_generator__(self):
        ...


class Generator(BaseGenerator):
    def __init__(self, x, y, length, batch_size=1, validate_ratio=0.0, shuffle=False, name='mnist_generator'):
            super().__init__(x, y, length, batch_size, shuffle, validate_ratio, name)

    def __generator__(self):
        x = []
        y = []
        for i in range(self.batch_size):
            x.append(self.x[self.train_indexes[self.train_index + i]])
            y.append(self.y[self.train_indexes[self.train_index + i]])
        x = np.asarray(x, np.float)
        x = np.expand_dims(x, 3)
        y = np.asarray(y, np.float)

        x /= 255.
        return x, y

    def __validate_generator__(self):
        x = []
        y = []
        for i in range(self.batch_size):
            x.append(self.x[self.validate_indexes[self.train_index + i]])
            y.append(self.y[self.validate_indexes[self.train_index + i]])
        x = np.asarray(x, np.float)
        x = np.expand_dims(x, 3)
        y = np.asarray(y, np.float)

        return x, y
