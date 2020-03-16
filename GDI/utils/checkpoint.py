from abc import ABCMeta, abstractmethod


class CheckPoint(metaclass=ABCMeta):
    # Main CheckPoint
    @abstractmethod
    def check_condition(self, result):
        ...

    @abstractmethod
    def do_action(self, x, y, model):
        ...


class MyCheckPoint(CheckPoint):
    def check_condition(self, result):
        return True

    def do_action(self, x, y, model):
        model.save("model.h5")

