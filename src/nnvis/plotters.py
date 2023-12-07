from typing import List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import os
import sys
dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.dirname(os.path.join(os.path.realpath(__file__), "../../")))

# from nnvis.src.nnvis.utils import rename, tensor_preproc
from utils import rename, tensor_preproc

class Plotter:
    def __init__(self, plot_path=None):
        self.__all__ = {
            self.layer_ridge_plot.__name__: self.layer_ridge_plot,
            self.layer_violin_plot.__name__: self.layer_violin_plot
        }
        if plot_path is None:
            self.save_path = os.path.join(dir_path, "../../vis")
        else:
            self.save_path = plot_path

    @rename("Violin_batch")
    def violin_batch_plot(self, df: pd.DataFrame, vis_name="", hscale=0.3):
        g = plt.figure()
        # g.suptitle("Avg distribution")
        ax = g.add_subplot()
        ax.set_xlabel("Distribution")
        ax.set_ylabel("Layer name")
        lcount = len(df.T)
        figheight = lcount * hscale
        if figheight * 100 > 2 ** 16:  # 2**16 is a hard limit in matplotlib
            split_step = int(np.floor((2 ** 16 / 100) / hscale))
            split = [a for a in range(0, lcount, split_step)]
            part_df = [df.T[i:i+split_step].T for i in split]
            for i, df_ in enumerate(part_df):
                g.set_figheight(len(df_.T) * hscale)
                sns.violinplot(data=df_, inner='points', orient='h',
                               gridsize=500, scale="width", width=1.5, cut=10)
                plt.tight_layout()
                plt.savefig(os.path.join(self.save_path, f"vis_act_{vis_name}_{i}.png"))
        else:
            sns.violinplot(data=df, inner='points', orient='h',
                           gridsize=500, scale="width", width=1, cut=10)

            g.set_figheight(lcount * hscale)
            plt.tight_layout()
            plt.savefig(os.path.join(self.save_path, f"vis_act_{vis_name}.png"))
        
        print(f"File saved at {self.save_path}")

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
        sns.violinplot(data=x, inner='points', orient='h',
                       gridsize=500, scale="width", width=1, cut=10)

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
