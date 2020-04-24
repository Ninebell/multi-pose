from keras.utils import plot_model
from keras.optimizers import RMSprop
from models import MultiPose, data_generator, save_heatmap
import numpy as np
from keras.layers import MaxPooling2D


def find_max_points(heat_map, ch):
    heat_map = heat_map[:,:,ch]
    # kernel = np.ones((3, 3), dtype=np.float) * 1/9.
    points=[]
    for r in range(1, heat_map.shape[0]-1):
        for c in range(1, heat_map.shape[1]-1):
            m_v = heat_map[r-1:r+2, c-1:c+2].max()
            v = heat_map[r,c]
            if v == m_v and v > 0.0:
                points.append((r,c))

    print(ch, len(points))
    # np.max()
    # t = heat_map[1:4,1:4] * kernel


    # equal_map = np.equal(t, heat_map)
    # return t



if __name__ == "__main__":
    ones = np.ones((15, 2))
    test = np.zeros((15, 2))
    for i in range(15):
        for j in range(2):
            test[i,j] = ((i+j) % 10+1)
            print(i,j,(i+1)% 10 + 1)
    print(test)
    eq = np.equal(ones, test)
    print(np.sum(eq))


if __name__ == "__main__2":

    length = 22420
    t = MultiPose(input_shape=(256, 256, 3), hournum=2)
    plot_model(t.model, to_file='model.png')
    optimzer = RMSprop(lr=2.5e-4)
    t.compile(optimizer=optimzer, loss='mse', metrics=['mae'])
    t.model.load_weights('dataset/mpii/result_paper/116/model.h5')

    for idx, value in enumerate(data_generator(50, 1, 1, shuffle=False)):
        save_heatmap(value[1][0][0], 'dataset/mpii/result_paper/{0}/heatmap_gt_{1}.png'.format(117, idx))

        result = t.model.predict(value[0])
        heatmaps = result[2]
        for i in range(17):
            find_max_points(value[1][0][0], i)
        limbs = result[3]
        # save_heatmap(result[0][0], 'dataset/mpii/result_paper/{0}/base_heatmap_{1}.png'.format(epoch, idx))
        # save_limb(result[1][0], 'dataset/mpii/result_paper/{0}/base_limb{1}.png'.format(epoch, idx))
        #
        save_heatmap(result[2][0], 'dataset/mpii/result_paper/{0}/heatmap_{1}.png'.format(117, idx))
        # save_limb(result[3][0], 'dataset/mpii/result_paper/{0}/limb_{1}.png'.format(epoch, idx))

