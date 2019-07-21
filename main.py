import os
import numpy as np
import random

from Orchestration.get_data import (
    get_train_data,
    piano_to_midi,
    inst_to_midi,
    cashe_data,
    get_str_data,
)
from Orchestration.model import MultipleRNN, EmbeddedRNN
from Orchestration import base_path

import music21

# path = "/Users/brianmodel/Desktop/gatech/VIP/DeepOrchestration/data/bouliane_aligned/0/Brahms_Symph4_iv(1-33)_ORCH+REDUC+piano_orch.mid"
# score = music21.converter.parse(path)

# print("DONE")
# X, y = get_train_data(source="hand_picked_Spotify", fix=False)
# print(y)
print("Loading data...")
X, y = get_str_data()
print("Data loaded successfully!")
model = EmbeddedRNN()
model.fit(X, y)
# print(y[10].keys())
# raise
# rnn = MultipleRNN()
# rnn.fit(X, y)
# data = X[10]
# print(y[10].keys())
# raise
# original = y[10]["Tuba"]
# inst_to_midi(original, "tuba_orig")
# piano_to_midi(data)
# rnn.predict(X[10], "Tuba")
