import os

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

dir_path = os.path.dirname(os.path.realpath(__file__))

def pdf_export(hook, plotter, fpath=None):
    if not fpath:
        fpath = os.path.join(dir_path, "../vis/pdfs")
    if not os.path.exists(fpath):
        os.mkdir(fpath)
    filepath = os.path.join(fpath, f"vis_{hook.name}-{plotter.__name__.lower()}.pdf")
    
    with PdfPages(filepath) as pdfp:
        for i, m in enumerate(hook.hook_data.keys()):
            plot = plotter(m, hook.hook_data[m])
            pdfp.savefig(plot)

def pdf_plot(plots, name="", fpath=None):
    if not fpath:
        fpath = os.path.join(dir_path, "../vis/pdfs")
    if not os.path.exists(fpath):
        os.mkdir(fpath)
    filepath = os.path.join(fpath, f"vis_{name}.pdf")
    with PdfPages(filepath) as pdfp:
        for plot in plots:
            pdfp.savefig(plot)