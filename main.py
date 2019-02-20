import os

from Orchestration.get_data import cashe_data
from Orchestration import data_path

cashe_data(os.path.join(data_path, 'bouliane_aligned'))