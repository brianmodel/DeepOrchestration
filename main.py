import os

from Orchestration.get_data import get_train_data, cashe_data
from Orchestration import data_path

data = get_train_data()
print(data)