import os
import numpy as np
import random

from Orchestration.get_data import get_train_data, orch_to_midi, piano_to_midi
from Orchestration.midi import read_midi, write_midi
from Orchestration import data_path, base_path

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.models import load_model


class MultipleRNN:
    def __init__(self):
        self.models = {}

    def fit(self, X, y):
        for inst in y[0].keys():
            y_inst = []
            for orch in y:
                y_inst.append(orch[inst])
            self.train(inst, X, y)

    # have an array of a single instrument and the piano roll. Train on that
    def train(self, inst, X, y):
        model = self._new_model_factory()
        print("NEW INSTRUMENT: ", inst)
        print("---------------------------------------------------------------")
        model.fit(X, y, epochs=500, verbose=1)
        self.models[inst] = model
        model.save("models/{}.h5".format(inst))

    def _new_model_factory(self):
        model = Sequential()
        model.add(LSTM(30, input_shape=(1, 128), return_sequences=True))
        model.add(Dense(30, activation="relu"))
        model.add(Dense(128, activation="linear"))
        model.compile(loss="mse", optimizer="adam")
        return model


# X, y = get_train_data()
# print(X, y)
# if not os.path.isfile(os.path.join(base_path, "lstm_model.h5")):
#     X = X[3]
#     y = y[3]
#     y = y[:-4]
#     X = X.reshape(X.shape[0], 1, 128)
#     y = y.reshape(y.shape[0], 1, 9472)

# X = X[3]
# y = y[3]
# y = y[:-4]
# preds = predict(X)
# piano_to_midi(X, os.path.join(base_path, "Orchestration/test.mid"))
# orch_to_midi(preds, os.path.join(base_path, "Orchestration/pred.mid"))
# orch_to_midi(y, os.path.join(base_path, "Orchestration/original_orch.mid"))


def predict(model, X):
    model = load_model("lstm_model.h5")
    X = X.reshape(X.shape[0], 1, 128)
    preds = model.predict(X)
    preds = preds.reshape(preds.shape[0], preds.shape[2])
    return preds
