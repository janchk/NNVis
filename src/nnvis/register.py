import torch.nn as nn


def hook_register(model: nn.Module, hook, exclude=[]):
    handles = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0 and module._get_name() not in exclude:
            handles.append(model.get_submodule(name).register_forward_hook(hook(name)))
    print(f"Hook '{hook.name}' registered!")
    return handles


def hook_unregister(handles):
    for handle in handles:
        handle.remove()
    print("Hook unregistered!")