import os
import numpy as np
import random

from Orchestration.get_data import (
    get_train_data,
    piano_to_midi,
    inst_to_midi,
    cashe_data,
)
from Orchestration.model import MultipleRNN
from Orchestration import base_path

# X, y = get_train_data(source="hand_picked_Spotify", fix=False)
X, y = get_train_data()
# print(y[10].keys())
# raise
rnn = MultipleRNN()
rnn.fit(X, y)
# data = X[10]
# print(y[10].keys())
# raise
# original = y[10]["Tuba"]
# inst_to_midi(original, "tuba_orig")
# piano_to_midi(data)
# rnn.predict(X[10], "Tuba")
