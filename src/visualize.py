import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, '.'))

import hooks
from exporter import pdf_export
from plotters import Plotter
from register import hook_register


class NVIS():
    def __init__(self,
                 plotter_name: str,
                 hook_names: list) -> None:

        for _hook in hook_names:
            if _hook not in hooks.__all__:
                raise NotImplementedError

        if len(hook_names) > 1:
            raise NotImplementedError(
                "Currently you can register only one hook at a time!")

        if plotter_name not in Plotter().__all__:
            raise NotImplementedError(
                f"Implemented plotters are {[p for p in Plotter().__all__.keys()]}!")

        self.hook_names = hook_names
        self.plotter = Plotter().__all__[plotter_name]
        self._hooks = [hooks.__dict__[name]() for name in self.hook_names]

    def __call__(self, model: torch.nn.Module) -> None:
        for _hook in self._hooks:
            hook_register(model, _hook)

    def export_pdf(self):
        plotter = self.plotter
        for _hook in self._hooks:
            pdf_export(_hook, plotter)
