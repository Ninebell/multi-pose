import utils.platform
from model.model_base import MNIST_Test
from utils.generator import Generator
from utils.checkpoint import MyCheckPoint
from utils.util import DataLoader
import numpy as np
import keras.models


if __name__ == "__main__":
    #   Set Data Loader
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
    x_test = np.reshape(x_test, (-1,28,28,1))

    x_test = np.asarray(x_test, np.float)

    x_test /= 255.

    # Generator Create
    gen = Generator(x_train, y_train, length=x_train.shape[0], validate_ratio=0.3 ,batch_size=60)

    # Checkpoint Create
    my = MyCheckPoint()

    # Model Create
    model = MNIST_Test((28,28,1))

    # Model Compile
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['acc'])
    model.summary()

    model.train(gen, steps=5000)

    ev1 = model.model.evaluate(x_test, y_test, verbose=2)
    print(ev1)

