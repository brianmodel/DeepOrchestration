from __future__ import print_function, division
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


import os
import numpy as np
from Orchestration.get_data import get_train_data, vect_to_midi, devectorize_orch
from Orchestration.midi import read_midi, write_midi
from Orchestration import data_path, base_path
import torch
import torch.nn as nn

import tensorflow as tf
import collections
import os
import argparse
import datetime as dt
import keras
import keras.backend as K

save_path = base_path + "/checkpoints"
# train()
#X, y = get_train_data()
# X = np.array(X[0])
# y = np.array(y[0])
from keras.models import Model
from keras.layers import Input, LSTM, Dense
import numpy as np
from Orchestration.get_data import get_train_data, devectorize_orch
from Orchestration.midi import read_midi, write_midi
from Orchestration import data_path, base_path
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from numpy import array
from keras.models import load_model


def predict(X):
    model = load_model("lstm_model.h5")
    X = X.reshape(X.shape[0], 1, 128)
    preds = model.predict(X)
    preds = preds.reshape(preds.shape[0], preds.shape[2])
    return preds


X, y = get_train_data()

if not os.path.isfile(os.path.join(base_path, "lstm_model.h5")):
    X = X[3]
    y = y[3]
    y = y[:-4]
    X = X.reshape(X.shape[0], 1, 128)
    y = y.reshape(y.shape[0], 1, 9472)

    model = Sequential()
    model.add(LSTM(30, input_shape=(1, 128), return_sequences=True))
    model.add(Dense(30, activation="relu"))
    model.add(Dense(9472, activation="linear"))
    model.compile(loss="mse", optimizer="adam")
    model.fit(X, y, epochs=500, verbose=1)
    model.save("lstm_model.h5")

X = X[4]
y = y[4]
preds = predict(X)
piano_to_midi(X, os.path.join(base_path, "Orchestration/test.mid"))
orch_to_midi(preds, os.path.join(base_path, "Orchestration/pred.mid"))
orch_to_midi(y, os.path.join(base_path, "Orchestration/original_orch.mid"))

