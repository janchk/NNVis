import os
import warnings

from models.mnist_simple_clamped_cnn import CLS
from visualize import NVIS

warnings.simplefilter(action='ignore', category=FutureWarning)
dir_path = os.path.dirname(os.path.realpath(__file__))

if __name__ == "__main__":
    cls = CLS()
    cls.eval()
    nvis = NVIS()
    nvis.set_model(cls)
    nvis.plot_weights_distributions()
    nvis.plot_random_activations_distributions((1,1,28,28))