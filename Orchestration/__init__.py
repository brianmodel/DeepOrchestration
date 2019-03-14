from os.path import dirname, abspath, join
import torch

base_path = dirname(dirname(abspath(__file__)))
data_path = join(dirname(dirname(abspath(__file__))), "data")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
