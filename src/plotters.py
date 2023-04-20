import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, '..'))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

from src.utils import rename



def tensor_preproc(data):
    if not isinstance(data, torch.Tensor):
        return data

    if len(data.shape) < 3:
        data = data.reshape(-1, 1)
    else:
        data = data.reshape(-1, data.shape[1])
    data = data.to(torch.device('cpu'))
    return data.detach()
        


class Plotter:
    def __init__(self):
        self.__all__ = {
            self.layer_ridge_plot.__name__: self.layer_ridge_plot,
            self.layer_violin_plot.__name__: self.layer_violin_plot
        }
    
    @rename("Violin")
    def layer_violin_plot(self, layer_name, layer_data):
        x = tensor_preproc(layer_data)

        g = plt.figure()
        if x.shape[-1] > 30:
            g.set_figheight(x.shape[-1] * 0.3)
        g.suptitle(f"Layer {layer_name}")
        ax = g.add_subplot()
        ax.set_xlabel("Distribution")
        ax.set_ylabel("Channel")
        sns.violinplot(data=x, inner='points', orient='h',  gridsize=500, scale="width", width=1, cut=10)

        return g

    @rename("Ridge")
    def layer_ridge_plot(self, layer_name, layer_data):
        x = tensor_preproc(layer_data)

        g = np.tile([a for a in range(x.shape[1])], x.shape[0])

        print(len(g))
        df = pd.DataFrame(dict(x=x.flatten(), g=g))
        # m = df.g.map(ord)
        # df["x"] += m

        # Initialize the FacetGrid object
        pal = sns.cubehelix_palette(x.shape[1], rot=-2.25, light=.7)
        g = sns.FacetGrid(df, row="g", hue="g", aspect=10,
                            height=1.0, palette=pal)
        # g = sns.FacetGrid(df, row="g", hue="g", aspect=15, height=.5, palette=pal)

        # Draw the densities in a few steps
        g.map(sns.kdeplot, "x",
                bw_adjust=.5, clip_on=False,
                fill=True, alpha=1.0, linewidth=0.5)
        g.map(sns.kdeplot, "x", clip_on=False, color="w", lw=0.5, bw_adjust=.5)

        # passing color=None to refline() uses the hue mapping
        g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)
        # g.refline(y=0, linewidth=2, linestyle="-", color=".5", clip_on=False)

        # Define and use a simple function to label the plot in axes coordinates

        def label(x, color, label):
            ax = plt.gca()
            ax.text(0, .3, label, fontweight="bold", color=color,
                    ha="left", va="center", transform=ax.transAxes)

        g.map(label, "x")

        # Set the subplots to overlap
        g.figure.subplots_adjust(top=1.0, bottom=-100.0, hspace=0.0)
        # g.figure.subplots_adjust(top=5, bottom=0, hspace=2)

        # Remove axes details that don't play well with overlap
        g.set_titles("")
        g.set(yticks=[], ylabel="", xlabel=f"layer {layer_name}")
        g.despine(bottom=True, left=True)
        g.fig.tight_layout(h_pad=0, w_pad=0)
        return g

if __name__ == "__main__":
    _vis = Plotter()
    rs = np.random.RandomState(1979)
    _x = rs.randn(800)
    _x = _x.reshape((-1, 20))
    plot = _vis.layer_violin_plot(None, _x)
    plot.savefig(os.path.join(dir_path, "../vis/test.png"))
