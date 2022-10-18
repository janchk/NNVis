import torch.nn as nn


def hook_register(model: nn.Module, hook):
    for c in model.named_children():
        model.get_submodule(c[0]).register_forward_hook(hook(c[0]))
    print(f"Hook {hook.name} registered!")
