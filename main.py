import os
import numpy as np
import random

from Orchestration.get_data import get_train_data, piano_to_midi, inst_to_midi
from Orchestration.model import MultipleRNN
from Orchestration import base_path

X, y = get_train_data()
rnn = MultipleRNN()
# rnn.fit(X, y)
data = X[10]
original = y[10]["Flute 1"]
inst_to_midi(original, "flute_orig")
# piano_to_midi(data)
# rnn.predict(X[10], "Flute1")
