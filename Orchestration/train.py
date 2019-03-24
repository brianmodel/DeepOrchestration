from Orchestration.get_data import get_train_data
from torch.nn import nn

def train():
    X, y = get_train_data()
    rnn = nn.RNN(input_size=1, hidden_size=100)
