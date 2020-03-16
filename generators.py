from GDI.utils.generator import BaseGenerator
import numpy as np
from PIL import Image
from utils import create_kernel


class PoseHeatmapGenerator(BaseGenerator):
    def __init__(self, x, y, length, batch_size=1, validate_ratio=0.0, shuffle=False, name='mnist_generator'):
        super().__init__(x, y, length, batch_size, shuffle, validate_ratio, name)

    def __make_heatmap(self, person):
        base = np.zeros((64,64,16*4))
        for points in person:
            for idx, point in enumerate(points):
                inner_idx = idx * 4
                base[:,:,inner_idx] = np.maximum(base[:,:,inner_idx], create_kernel(self.shape, point, 9))
                base[:,:,inner_idx+1] = np.maximum(base[:,:,inner_idx+1], create_kernel(self.shape, point, 9))
                base[:,:,inner_idx+2] = np.maximum(base[:,:,inner_idx+2], create_kernel(self.shape, point, 9))
                base[:,:,inner_idx+3] = np.maximum(base[:,:,inner_idx+3], create_kernel(self.shape, point, 9))

        # print(x, len(x))
        # print(y, y.shape)

        return base

    def __generator__(self):
        x = []
        y = []
        for i in range(self.batch_size):
            x.append(Image.open(self.x[self.train_indexes[self.train_index + i]]))
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
