import torch.nn as nn


def hook_register(model: nn.Module, hook):
    handles = []
    for c in model.named_children():
        handles.append(model.get_submodule(c[0]).register_forward_hook(hook(c[0])))
    print(f"Hook '{hook.name}' registered!")
    return handles


def hook_unregister(handles):
    for handle in handles:
        handle.remove()
    print("Hook unregistered!")
