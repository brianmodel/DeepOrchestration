import os
import numpy as np
import random

from Orchestration.get_data import get_train_data
from Orchestration.model import MultipleRNN
from Orchestration import base_path

X, y = get_train_data()
rnn = MultipleRNN()
rnn.fit(X, y)
