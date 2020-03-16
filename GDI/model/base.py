from tqdm import tqdm as tqdm
from keras.models import Model
from abc import ABCMeta, abstractmethod
from keras.layers import Input
from ..utils.generator import BaseGenerator
from ..utils.checkpoint import CheckPoint
import numpy as np


class BaseModel(metaclass=ABCMeta):
    def __init__(self, input_shape, name="Base"):
        self.name = name
        self.input_shape = input_shape
        self.input_layer = None
        self.output_layer = None
        self.model = None
        self.build()

    def output_shape(self):
        return self.output_layer.output_shape

    def __call__(self, input_model):
        return self.model(input_model)

    def build(self):
        self.input_layer = Input(self.input_shape)
        self.output_layer = self.__build_model__(self.input_layer)
        self.model = Model(inputs=self.input_layer, outputs=self.output_layer, name=self.name)

    def compile(self, optimizer, loss=None, metrics=None, loss_weights=None, sample_weight_mode=None, weighted_metrics=None, target_tensors=None):
        self.model.compile(optimizer,loss,metrics,loss_weights,sample_weight_mode,weighted_metrics,target_tensors)

    def summary(self):
        self.model.summary()

    def __train_on_batch(self, x, y):
        return self.model.train_on_batch(x, y)

    def __print_result(self, result, name):
        line0 = "#"

        line1 = "#"
        line2 = "#"
        line3 = "#"
        for idx, metric in enumerate(self.model.metrics_names):
            line0 += "#"*22
            line1 += " {0:^20s} ".format(name)
            line2 += " {0:^20s} ".format(metric)
            line3 += " {0:^20f} ".format(result[idx])
        line0 += "#"
        line1 += "#"
        line2 += "#"
        line3 += "#"
        print(line0)
        print(line1)
        print(line2)
        print(line3)
        print(line0, '\n')

    def train(self, train_generator, steps=1, epochs=1, check_point=None):
        assert isinstance(train_generator, BaseGenerator), 'non generator'
        if check_point is not None:
            assert type(check_point) == list, 'check_point must be list of CheckPoint'
            for ch in check_point:
                assert isinstance(ch, CheckPoint), 'check_point element must be CheckPoint'

        metric_len = len(self.model.metrics_names)

        for epoch in range(epochs):
            print("Epoch {0:4d}/{1:4d}".format(epoch+1,epochs))
            train_result = np.zeros(metric_len, np.float)
            for step in tqdm(range(steps)):
                x, y = train_generator.generator()
                tep = self.__train_on_batch(x,y)
                train_result += tep

            train_result /= steps

            self.__print_result(train_result, name="train")

            self.__print_result(self.evaluate(train_generator), name="evaluate")


            if check_point is not None:
                for ch in check_point:
                    if ch.check_condition(train_result):
                        ch.do_action(x, y, self.model)


    def __evaluate(self, x, y):
        return self.model.evaluate(x, y, verbose=0)

    def evaluate(self, train_generator):
        metric_len = len(self.model.metrics_names)
        validate_result = np.zeros(metric_len, np.float)
        steps = train_generator.validate_length()
        for step in tqdm(range(steps)):
            x, y = train_generator.validate_generator()
            tep = self.__evaluate(x, y)
            validate_result += tep

        validate_result /= steps
        return validate_result

    @abstractmethod
    def __build_model__(self, input_layer):
        ...

