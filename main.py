import os
import numpy as np
import random

import torch
from torch import nn
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from Orchestration.get_data import get_train_data, orch_to_midi, devectorize_orch
from Orchestration.midi import read_midi, write_midi
from Orchestration import data_path, base_path
from Orchestration.train import train
from Orchestration.model import RNN

X, y = get_train_data()

def random_train_example():
    index = int(len(X) * random.random())
    return torch.as_tensor(X[index]).float(), torch.as_tensor(y[index])

n_inputs = 128
n_hidden = 128
n_outputs = 74

rnn = RNN(n_inputs, n_hidden, n_outputs)
criterion = nn.NLLLoss()
learning_rate = .005

def train(piano_roll, orchestration):
    hidden = rnn.initHidden()
    rnn.zero_grad()
    for i in range(piano_roll.size()[0]):
        reshaped = piano_roll[i].unsqueeze(0)
        output, hidden = rnn(reshaped, hidden)
    
    print(output.size())
    print(orchestration.size())
    loss = criterion(output, orchestration)
    loss.backward()

    for p in rnn.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.item()

plot_every = 1000
n_iters = 100000

for iter in range(1, n_iters + 1):
    X_train, y_train = random_train_example()
    output, loss = train(X_train, y_train)
    current_loss += loss

    # Add current loss avg to list of losses
    if iter % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        current_loss = 0

plt.figure()
plt.plot(all_losses)
