import collections


class ModuleHook:
    def __init__(self) -> None:
        self._module_name = None
        self._module_hook = None
        self.name = None
        self.hook_data = collections.defaultdict(dict)
        self.hook_count = {}

    def __call__(self, layer_name=None, *args):
        raise NotImplementedError("You need to implement __call__ method!")

    def __repr__(self) -> str:
        return str(self.hook_data)


class InputDataHook(ModuleHook):
    def __init__(self) -> None:
        super().__init__()
        self.name = "input"

    def __call__(self, layer_name):
        def _hook(model, input, output):
            try:
                self.hook_data[layer_name] = input.detach()
            except AttributeError:
                self.hook_data[layer_name] = input[0]
        return _hook


class OutputDataHook(ModuleHook):
    def __init__(self, avg_channels=False) -> None:
        super().__init__()
        self.name = "output"

    def __call__(self, layer_name):
        if layer_name not in self.hook_count:
            self.hook_count[layer_name] = 0
        def _hook(model, input, output):
            if self.hook_count[layer_name] == 0: 
                try:
                    self.hook_data[layer_name] = output.detach()
                except AttributeError:
                    self.hook_data[layer_name] = output[0]
                self.hook_count[layer_name] += 1
            elif self.hook_count[layer_name] > 0:
                try:
                    self.hook_data[layer_name] += (output.detach() - self.hook_data[layer_name]) / self.hook_count[layer_name]
                except AttributeError:
                    self.hook_data[layer_name] += (output[0] - self.hook_data[layer_name]) / self.hook_count[layer_name]
                self.count += 1
            else:
                raise ValueError

        return _hook
