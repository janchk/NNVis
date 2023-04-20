import os, sys; sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from register import hook_register, hook_unregister
from plotters import Plotter
from exporter import pdf_export, pdf_plot
import hooks

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch


class NVIS():
    def __init__(self) -> None:
        self.model: torch.nn.Module = None
        self.plotter = Plotter()
        self.plots = []
        self.hook = None

    def set_model(self, model: torch.nn.Module) -> None:
        self.model = model

    def plot_weights_distributions(self, name=""):
        self.plots = []
        def filter(name): return name.find(".bn") == -1 and "bias" not in name

        for _name, weights in self.model.named_parameters():
            if filter(_name):
                plot = self.plotter.layer_violin_plot(_name, weights)
                self.plots.append(plot)

        pdf_plot(self.plots, f"weights_{name}")

    def plot_activations_distributions(self, input_shape, name=""):
        self.plots = []
        data = torch.rand(input_shape)
        self.hook = hooks.OutputHook()
        handles = hook_register(self.model, self.hook)

        self.model(data)

        for _name in self.hook.hook_data.keys():
            plot = self.plotter.layer_violin_plot(_name, self.hook.hook_data[_name])
            self.plots.append(plot)

        hook_unregister(handles)

        pdf_plot(self.plots, f"activations_{name}")

    # def export_pdf(self):
        # plotter = self.plotter
        # for _hook in self._hooks:
            # pdf_export(_hook, plotter)
