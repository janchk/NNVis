import torch.nn as nn
from typing import List


def hook_register(model: nn.Module, hook, includes: List[None | nn.Module] = [], exclude: List[None | str] = []):
    handles = []
    # includes = [nn.Conv2d]
    for name, module in model.named_modules():
        if len(includes) > 0:
            if issubclass(module.__class__, tuple(includes)):
                handles.append(model.get_submodule(name).register_forward_hook(hook(name)))
        elif len(list(module.children())) == 0 and module._get_name() not in exclude:
            handles.append(model.get_submodule(name).register_forward_hook(hook(name)))
    print(f"Hook '{hook.name}' registered!")
    return handles


def hook_unregister(handles):
    for handle in handles:
        handle.remove()
    print("Hook unregistered!")
