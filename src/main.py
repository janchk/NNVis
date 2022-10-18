import torch
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
from models.mnist_simple_clamped_cnn import CLS

from register import hook_register
from hooks import InputHook, OutputHook
from exporter import pdf_export

if __name__ == "__main__":
    cls = CLS()
    cls.eval()
    in_hook = InputHook()
    out_hook = OutputHook()

    hook_register(cls, out_hook)
    hook_register(cls, in_hook)


    data = torch.rand(1, 1, 28, 28)
    cls(data)
    pdf_export(out_hook)