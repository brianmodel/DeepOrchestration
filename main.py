import os

from Orchestration.get_data import get_train_data, cashe_data
from Orchestration import data_path

# cashe_data(os.path.join(data_path, 'bouliane_aligned'))
data = get_train_data()[2][0]
print(data)
