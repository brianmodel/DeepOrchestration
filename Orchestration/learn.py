import tensorflow as tf
from Orchestration.get_data import get_train_data
from Orchestration.model import create_model
import numpy as np


def learn():
    X, y = get_train_data()
    # X = X[0].reshape(1, 110, 128)
    X = X[0]
    y = np.array(y[0])
    print(X.shape)
    model = create_model()
    model.fit(X, y, epochs=3)
