import os
import numpy as np
import random

from Orchestration.get_data import (
    get_train_data,
    orch_to_midi,
    devectorize_orch,
    piano_to_midi,
)
from Orchestration.midi import read_midi, write_midi
from Orchestration import data_path, base_path
from Orchestration.learn import learn

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from numpy import array
from keras.models import load_model


def predict(X):
    model = load_model("lstm_model.h5")
    X = X.reshape(X.shape[0], 1, 128)
    preds = model.predict(X)
    preds = preds.reshape(preds.shape[0], preds.shape[2])
    return preds

def train(trainModel, X, y, num_epochs=50):
    print("training " + str(num_epochs) + " more epochs")
    X = X.reshape(X.shape[0], 1, 128)
    y = y.reshape(y.shape[0], 1, 9472)
    trainModel.fit(X, y, epochs=num_epochs, verbose=1)



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
    model.add(BatchNormalization())
    model.add(LSTM(30, input_shape=(1, 128), return_sequences=True))
    # model.add(MaxPooling1D(pool_size=10))
    model.add(Dense(9472, activation="linear"))
    model.add(Dropout(rate=.25))
    model.compile(loss="mse", optimizer="adam")
    # model.fit(X, y, epochs=1000, verbose=1)
    model.save("lstm_model.h5")


# model = load_model("lstm_model.h5")
# for i in range(10):
#     for i in range(min(len(X), len(y))):
#         print("---------------")
#         print("training on song: " + str(i))
#         tempX = X[i]
#         tempY = y[i]
#         print(len(tempX))
#         print(len(tempY))
#         if len(tempY) < len(tempX):
#             tempX = tempX[:len(tempY)-len(tempX)]
#         if len(tempY) > len(tempX):
#             tempY = tempY[:len(tempX)-len(tempY)]
#         print(len(tempX))
#         print(len(tempY))
#         train(model, tempX, tempY, 15)

# model.save("lstm_model.h5")


X = X[7]
y = y[7]
preds = predict(X)
piano_to_midi(X, os.path.join(base_path, "Orchestration/test.mid"))
orch_to_midi(preds, os.path.join(base_path, "Orchestration/pred.mid"))
orch_to_midi(y, os.path.join(base_path, "Orchestration/original_orch.mid"))


