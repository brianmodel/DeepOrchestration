import os
import numpy as np
from Orchestration.get_data import get_train_data, orch_to_midi, devectorize_orch
from Orchestration.midi import read_midi, write_midi
from Orchestration import data_path, base_path
from Orchestration.train import train


train()