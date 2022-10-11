import torch
import collections

class Hooks():
    def __init__(self) -> None:
        self.activation = collections.defaultdict(dict)
        self.input = collections.defaultdict(dict)

    def _set_data(self, type, name, data):
        self.__dict__[type][name]['data'] = data
        self.__dict__[type][name]['shape'] = data.shape
        # self.__dict__[type][name]['layer_type']

    def get_activation(self, name):
        def hook(model, input, output):
            self._set_data('activation', name, output.detach())
        return hook

    def get_input(self, name):
        def hook(model, input, output):
            try:
                self._set_data('input', name, input.detach())
            except AttributeError:
                self._set_data('input', name, input[0])
        return hook
