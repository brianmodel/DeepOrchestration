import os
import numpy as np
import random

from Orchestration.get_data import get_train_data, orch_to_midi, piano_to_midi
from Orchestration.midi import read_midi, write_midi
from Orchestration import data_path, base_path

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Dropout
from keras.models import load_model
from keras.callbacks import ModelCheckpoint


class MultipleRNN:
    def __init__(self):
        self.models = {}

    def fit(self, X, y):
        for inst in y[0].keys():
            y_inst = []
            for orch in y:
                y_inst.append(orch[inst])
            self.train(inst, X, y_inst)

    def train(self, inst, X, y):
        model = self._new_model_factory()
        epochs = 100
        print("NEW INSTRUMENT: ", inst)
        print(
            "--------------------------------------------------------------------------------"
        )
        '''
        checkpoint = ModelCheckpoint(
            "models/checkpoints/" + inst.strip() + "-checkpoint-{epoch:02d}.hdf5",
            monitor="val_acc",
            verbose=1,
            save_best_only=True,
            mode="max",
        )
        callbacks_list = [checkpoint]
        '''
        model.fit_generator(
            MultipleRNN.generator_sample(X, y),
            steps_per_epoch=300,
            epochs=500,
            verbose=1,
        )
        self.models[inst] = model
        model.save(base_path + "/Orchestration/models/{}.h5".format(inst.replace(' ', '')))

    @staticmethod
    def generator_sample(X, y):
        while True:
            index = int(random.random() * len(X))
            X_train = X[index].reshape(X[index].shape[0], 1, 128)
            y_train = y[index].reshape(y[index].shape[0], 1, 128)
            yield X_train, y_train

    def _new_model_factory(self):
        model = Sequential()
        model.add(LSTM(30, input_shape=(1, 128), return_sequences=True))
        model.add(Dense(30, activation="relu"))
        model.add(Dense(128, activation="linear"))
        model.add(Dropout(rate=0.5))
        model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
        return model


def predict(model, X):
    model = load_model("lstm_model.h5")
    X = X.reshape(X.shape[0], 1, 128)
    preds = model.predict(X)
    preds = preds.reshape(preds.shape[0], preds.shape[2])
    return preds
