import os, sys
sys.path.append(os.path.dirname(os.path.realpath(__file__)))
from nnvis.register import hook_register, hook_unregister
from nnvis.plotters import Plotter
from nnvis.exporter import pdf_export, pdf_plot 
from nnvis.utils import tensor_preproc, tensor_sample_preproc
from nnvis import hooks
dir_path = os.path.dirname(os.path.realpath(__file__))
import matplotlib.pyplot as plt
import pandas as pd
import torch


class NVIS():
    def __init__(self, name="") -> None:
        self.model: torch.nn.Module = None
        self.out_path  = "vis"
        self.plotter = Plotter(plot_path=self.out_path)
        self.plots = []
        self.hook = None
        self.handles = None
        self.vis_name = name

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

    def track_activations(self, include_layers=[], exclude_layers=[]):
        self.hook = hooks.OutputDataHook()
        self.handles = hook_register(self.model, self.hook, include_layers, exclude_layers)
    
    def print_batch_dataframe(self, sampled=False):
        lnames = [n for n in self.hook.hook_data.keys()]
        if sampled:
            ldata = [tensor_sample_preproc(self.hook.hook_data[_name]).numpy() for _name in lnames]
        else:
            ldata = [tensor_preproc(self.hook.hook_data[_name], True).numpy() for _name in lnames]

        df = pd.DataFrame(ldata).T
        df.columns = [f"{name} [{self.model.get_submodule(name).__class__.__name__}]" for name in lnames] # extend text representation of a layer

        df.to_csv(os.path.join(f"{self.out_path}", f"dataframe_{self.vis_name}.csv"))
        
    
    def batch_plot_tracking_activations(self, name=None):
        if name is not None:
            self.vis_name = name
        lnames = [n for n in self.hook.hook_data.keys()]
        
        ldata = [tensor_sample_preproc(self.hook.hook_data[_name]).numpy() for _name in lnames]

        df = pd.DataFrame(ldata).T
        df.columns = [f"{name} [{self.model.get_submodule(name).__class__.__name__}]" for name in lnames] # extend text representation of a layer

        self.plotter.violin_batch_plot(df, self.vis_name)

        hook_unregister(self.handles)

    
    def plot_tracked_activations(self):
        self.plots = []

        for _name in self.hook.hook_data.keys():
            plot = self.plotter.layer_violin_plot(_name, self.hook.hook_data[_name])
            self.plots.append(plot)
            plt.close()
        
        hook_unregister(self.handles)
        pdf_plot(self.plots, f"activations_{self.vis_name}")

        
    def plot_avg_batch_activations_distributions(self, input_batch, name=""):
        self.plots = []
        self.hook = hooks.OutputDataHook()
        self.handles = hook_register(self.model, self.hook)

        for item in iter(input_batch):
            self.model(item)
        
        for _name in self.hook.hook_data.keys():
            plot = self.plotter.layer_violin_plot(_name, self.hook.hook_data[_name])
            self.plots.append(plot)

        hook_unregister(self.handles)
        plt.savefig("activations_new.pdf")
        # pdf_plot(self.plots, f"activations_{name}")

    def plot_random_activations_distributions(self, input_shape):
        self.plots = []
        data = torch.rand(input_shape)
        self.hook = hooks.OutputDataHook()
        handles = hook_register(self.model, self.hook)

        self.model(data)

        for _name in self.hook.hook_data.keys():
            plot = self.plotter.layer_violin_plot(_name, self.hook.hook_data[_name])
            self.plots.append(plot)

        hook_unregister(handles)

        pdf_plot(self.plots, f"activations_{self.vis_name}")


    # def export_pdf(self):
        # plotter = self.plotter
        # for _hook in self._hooks:
            # pdf_export(_hook, plotter)
