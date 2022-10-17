import torch
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
from models.mnist_simple_clamped_cnn import CLS

from register import hook_register
from hooks import InputHook, OutputHook
from exporter import pdf_export

if __name__ == "__main__":
    # vis = VIS()
    cls = CLS()
    cls.eval()
    # _hooks = Hooks()
    # out_hook = _hooks.get_activation
    in_hook = InputHook()

    for c in cls.named_children():
        cls.get_submodule(c[0]).register_forward_hook(in_hook(c[0]))

    data = torch.rand(1, 1, 28, 28)
    cls(data)
    pdf_export(in_hook)
    # for m in in_hook.hook_data.keys():
    #     fig = vis.layer_ridge_plot(m, in_hook.hook_data[m])
    #     fig.savefig(os.path.join(dir_path,f"../vis/{in_hook.name}_{m}_plot.png"))
