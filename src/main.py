import torch

from models.mnist_simple_clamped_cnn import CLS
from visualize import VIS
from register import hook_register
from hooks import Hooks

if __name__ == "__main__":
    vis = VIS()
    cls = CLS()
    cls.eval()
    _hooks = Hooks()
    out_hook = _hooks.get_activation
    in_hook = _hooks.get_input

    for c in cls.named_children():
        cls.get_submodule(c[0]).register_forward_hook(in_hook(c[0]))

    data = torch.rand(1, 1, 28, 28)
    cls(data)
    for m in _hooks.input.keys():
        fig = vis.layer_ridge_plot(m, _hooks.input[m]['data'])
        fig.savefig(f"../vis/{m}_plot.png")
