from Orchestration.learn import train
import os
import numpy as np
from Orchestration.get_data import get_train_data, vect_to_midi, devectorize_orch
from Orchestration.midi import read_midi, write_midi
from Orchestration import data_path, base_path


# train()
X, y = get_train_data()
print(X)
# orch = devectorize_orch(y[0])
# print(orch)
vect_to_midi(y[0])
# vect_to_midi(X)
# write_midi.write_midi({'Kboard': X[0]}, 8, os.path.join(base_path, "Orchestration/temp.mid"))

