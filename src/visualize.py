import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import torch

import os 
dir_path = os.path.dirname(os.path.realpath(__file__))


class VIS():
    def __init__(self):
        pass

    @staticmethod
    def _tensor_preproc(data):
        if len(data.shape) < 3:
            data = data.reshape(-1, 1)
        else:
            data = data.reshape(-1, data.shape[1])
        return data.detach()

    def layer_ridge_plot(self, layer_name, layer_data):
        if isinstance(layer_data, torch.Tensor):
            x = self._tensor_preproc(layer_data)
        else:
            x = layer_data    
        g = np.tile([a for a in range(x.shape[1])], x.shape[0])

        print(len(g))
        df = pd.DataFrame(dict(x=x.flatten(), g=g))
        # m = df.g.map(ord)
        # df["x"] += m

        # Initialize the FacetGrid object
        pal = sns.cubehelix_palette(x.shape[1], rot=-1.25, light=.7)
        g = sns.FacetGrid(df, row="g", hue="g", aspect=12,
                          height=.5, palette=pal)
        # g = sns.FacetGrid(df, row="g", hue="g", aspect=15, height=.5, palette=pal)

        # Draw the densities in a few steps
        g.map(sns.kdeplot, "x",
              bw_adjust=.5, clip_on=False,
              fill=True, alpha=1.0, linewidth=0.5)
        g.map(sns.kdeplot, "x", clip_on=False, color="w", lw=0.1, bw_adjust=.5)

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
        g.figure.subplots_adjust(top=0.01, bottom=-0.6, hspace=0.10)

        # Remove axes details that don't play well with overlap
        g.set_titles("")
        g.set(yticks=[], ylabel="", xlabel=f"{layer_name}")
        g.despine(bottom=True, left=True)
        # g.fig.tight_layout(w_pad = 100)
        return g


if __name__ == "__main__":
    _vis = VIS()
    rs = np.random.RandomState(1979)
    _x = rs.randn(800)
    _x = _x.reshape((-1, 20))
    plot = _vis.layer_ridge_plot(None, _x)
    plot.savefig(os.path.join(dir_path, "../vis/test.png"))
