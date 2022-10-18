import os
import warnings

import torch

from models.mnist_simple_clamped_cnn import CLS
from visualize import NVIS

warnings.simplefilter(action='ignore', category=FutureWarning)
dir_path = os.path.dirname(os.path.realpath(__file__))

if __name__ == "__main__":
    cls = CLS()
    cls.eval()
    nvis = NVIS("", ["InputHook"])
    nvis(cls)

    data = torch.rand(1, 1, 28, 28)
    cls(data)

    nvis.export_pdf()
