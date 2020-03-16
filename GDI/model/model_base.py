from keras.layers import Dense, Flatten, Dropout, BatchNormalization
from model.base import BaseModel


class MNIST_Test(BaseModel):
    def __init__(self, input_shape, name="ModelA"):
        super().__init__(input_shape, name)

    def __build_model__(self, input_layer):
        flat = Flatten()(input_layer)
        dense = Dense(128, activation='relu')(flat)
        batch = BatchNormalization()(dense)
        dense1 = Dense(10, activation='softmax')(batch)
        # dense2 = Dense(10, activation='softmax')(dropout)
        return dense1




