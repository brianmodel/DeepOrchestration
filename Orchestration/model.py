import os
import numpy as np
import random

from Orchestration.get_data import (
    get_train_data,
    inst_to_midi,
    piano_to_midi,
    orch_to_midi,
)
from Orchestration.midi import read_midi, write_midi
from Orchestration import data_path, base_path, instruments

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Dropout
from keras.models import load_model
from keras.callbacks import ModelCheckpoint

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, Sequential
from keras.layers import (
    GRU,
    Input,
    Dense,
    TimeDistributed,
    Activation,
    RepeatVector,
    Bidirectional,
    Dropout,
    LSTM,
)
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy

from gensim.models import Word2Vec
from embedding.utils import embedded

"""
TODO

Create different models:
encoder-decoder RNN (https://arxiv.org/abs/1811.12408, https://towardsdatascience.com/representing-music-with-word2vec-c3c503176d52, https://arxiv.org/abs/1706.09088)
RNN-RBM (http://danshiebler.com/2016-08-17-musical-tensorflow-part-two-the-rnn-rbm/)
RNN-DBN (similar to RNN-RBM)
"""


class MultipleRNN:
    def __init__(self):
        self.models = {}

    def fit(self, X, y):
        for inst in y[0].keys():
            y_inst = []
            for orch in y:
                y_inst.append(orch[inst])
            self.train(inst, X, y_inst)

    def predict(self, X, inst=None, save=True):
        if inst != None:
            model = load_model(base_path + "/Orchestration/models/{}.h5".format(inst))
            X = X.reshape(X.shape[0], 1, 128)
            preds = model.predict(X)
            preds = preds.reshape(preds.shape[0], preds.shape[2])
        else:
            pass
        if save:
            inst_to_midi(preds, inst)
        else:
            return preds

    def train(self, inst, X, y):
        model = MultipleRNN._new_model_factory()
        epochs = 100
        print("NEW INSTRUMENT: ", inst)
        print(
            "--------------------------------------------------------------------------------"
        )
        model.fit_generator(
            MultipleRNN.generator_sample(X, y),
            steps_per_epoch=300,
            epochs=500,
            verbose=1,
        )
        self.models[inst] = model
        model.save(
            base_path + "/Orchestration/models/{}.h5".format(inst.replace(" ", ""))
        )

    @staticmethod
    def generator_sample(X, y):
        while True:
            index = int(random.random() * len(X))
            X_train = X[index].reshape(X[index].shape[0], 1, 128)
            y_train = y[index].reshape(y[index].shape[0], 1, 128)
            yield X_train, y_train

    @staticmethod
    def _new_model_factory():
        # TODO Play around with values
        model = Sequential()
        model.add(LSTM(30, input_shape=(1, 128), return_sequences=True))
        model.add(Dense(30, activation="relu"))
        model.add(Dense(128, activation="linear"))
        # Lower rate
        model.add(Dropout(rate=0.5))
        model.compile(loss="mse", optimizer="adam", metrics=["accuracy"])
        return model

    @staticmethod
    def train_classifier():
        X, y = get_train_data(fix=False)
        classifier = ClassifierRNN()
        classifier.fit(X, y)


class EmbeddedRNN:
    def __init__(self):
        self.models = {}
        self.embedding = Word2Vec.load("word2vec_nooctive_enharmonic.model")

    def fit(self, X, y):
        for inst in instruments:
            X_inst = []
            y_inst = []
            for i in range(len(y)):
                orch = y[i]
                if inst in orch:
                    X_inst.append(X[i])
                    y_inst.append(orch[inst])
            self.train(inst, X_inst, y_inst)

    def predict(self, X, inst=None, save=True):
        pass

    def train(self, inst, X, y):
        model = EmbeddedRNN._new_model_factory()
        epochs = 100
        print("NEW INSTRUMENT: ", inst)
        print(
            "--------------------------------------------------------------------------------"
        )
        model.fit_generator(
            self.generator_sample(X, y), steps_per_epoch=300, epochs=500, verbose=1
        )
        self.models[inst] = model
        model.save(
            base_path
            + "/Orchestration/models/embedded/{}.h5".format(inst.replace(" ", ""))
        )

    def generator_sample(self, X, y):
        while True:
            index = int(random.random() * len(X))
            X_train = X[index]
            y_train = y[index].reshape(y[index].shape[0], 1, 128)
            yield X_train, y_train

    @staticmethod
    def _new_model_factory():
        learning_rate = 0.005
        model = Sequential()
        model.add(GRU(256, input_shape=(1, 300), return_sequences=True))
        model.add(TimeDistributed(Dense(1024, activation="relu")))
        model.add(Dropout(0.5))
        model.add(TimeDistributed(Dense(128, activation="softmax")))

        # Compile model
        model.compile(
            loss=categorical_crossentropy,
            optimizer=Adam(learning_rate),
            metrics=["accuracy"],
        )
        return model

        # # Encoder
        # model.add(Bidirectional(GRU(128)))
        # model.add(RepeatVector(output_sequence_length))
        # # Decoder
        # model.add(Bidirectional(GRU(128, return_sequences=True)))
        # model.add(TimeDistributed(Dense(512, activation='relu')))
        # model.add(Dropout(0.5))
        # model.add(TimeDistributed(Dense(french_vocab_size, activation='softmax')))
        # model.compile(loss=sparse_categorical_crossentropy,
        #             optimizer=Adam(learning_rate),
        #             metrics=['accuracy'])


class ClassifierRNN:
    def __init__(self):
        pass

    def fit(self, X, y):
        self.X = X
        insts = set()
        for orch in y:
            for inst in orch.keys():
                insts.add(inst)
        insts = list(insts)


def predict(model, X):
    model = load_model("lstm_model.h5")
    X = X.reshape(X.shape[0], 1, 128)
    preds = model.predict(X)
    preds = preds.reshape(preds.shape[0], preds.shape[2])
    return preds
