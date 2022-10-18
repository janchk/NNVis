import collections


class ModuleHook:
    def __init__(self) -> None:
        self._module_name = None
        self._module_hook = None
        self.name = None
        self.hook_data = collections.defaultdict(dict)

    def __call__(self, layer_name=None, *args):
        raise NotImplementedError("You need to implement __call__ method!")

    def __repr__(self) -> str:
        return str(self.hook_data)


class InputHook(ModuleHook):
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


class OutputHook(ModuleHook):
    def __init__(self) -> None:
        super().__init__()
        self.name = "output"

    def __call__(self, layer_name):
        def _hook(model, input, output):
            try:
                self.hook_data[layer_name] = output.detach()
            except AttributeError:
                self.hook_data[layer_name] = output[0]
        return _hook
