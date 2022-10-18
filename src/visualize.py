import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import hooks
from register import hook_register

dir_path = os.path.dirname(os.path.realpath(__file__))


class NVIS():
    def __init__(self, model: torch.nn.Module,
                 plotter_name: str,
                 hook_names: list) -> None:
        for _hook in hook_names:
            if _hook not in hooks.__all__:
                raise NotImplementedError
        
        self.hook_names = hook_names
        self.plotter_name = plotter_name


    def __call__(self, model: torch.nn.Module) -> None:
        for hook_name in self.hook_names:
            hook_register(model, hooks.__dict__[hook_name])
