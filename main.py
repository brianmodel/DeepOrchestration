import os
import numpy as np
import random

from Orchestration.get_data import get_train_data, orch_to_midi, devectorize_orch
from Orchestration.midi import read_midi, write_midi
from Orchestration import data_path, base_path
from Orchestration.learn import learn
from seq2seq.models import SimpleSeq2Seq

X, y = get_train_data()
# print(X[0].shape)
# print(y[0][0].shape)

model = SimpleSeq2Seq(
    input_shape=(1, 111),
    hidden_dim=10,
    output_length=111,
    # output_dim=(74, 111, 128),
    output_dim=9472,
    depth=(4, 5),
)
X[0] = np.append(X[0], np.zeros(128))
deep_y = [[] for i in range(len(y[0][0]))]
for inst in y[0]:
    for j in range(len(inst)):
        deep_y[j].extend(inst[j])
deep_y = np.array(deep_y)

model.compile(loss="mse", optimizer="rmsprop")
model.fit(X[0].reshape(1, 111, 128), deep_y, verbose=1)
model.save_weights("model_weights.h5")

# learn()
