import os

import torch

from exporter import pdf_export
from hooks import InputHook, OutputHook
from models.mnist_simple_clamped_cnn import CLS
from plotters import Plotter
from register import hook_register

dir_path = os.path.dirname(os.path.realpath(__file__))

if __name__ == "__main__":
    cls = CLS()
    cls.eval()
    in_hook = InputHook()
    out_hook = OutputHook()
    plotter = Plotter().layer_ridge_plot

    hook_register(cls, out_hook)
    hook_register(cls, in_hook)

    data = torch.rand(1, 1, 28, 28)
    cls(data)
    pdf_export(out_hook, plotter)

