import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from visualize import VIS

def pdf_export(hook, filepath=None):
    vis = VIS()
    filepath = f"vis_{hook.name}-hook.pdf" if not filepath else filepath
    with PdfPages(filepath) as pdfp:
        for i, m in enumerate(hook.hook_data.keys()):
            # fig = plt.figure(i)
            plot = vis.layer_ridge_plot(m, hook.hook_data[m])
            pdfp.savefig(plot.fig)