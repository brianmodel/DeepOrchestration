import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, CuDNNLSTM, RepeatVector


def create_model():
    model = Sequential()
    # Hard coding in numbers right now, not too good
    model.add(Dense(110, input_shape=[1, 128]))
    model.add(RepeatVector(74))
    model.add(LSTM(128, activation="relu"))
    model.add(LSTM(128, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(74, activation="softmax"))
    opt = tf.keras.optimizers.Adam(lr=1e-3, decay=1e-5)
    model.compile(
        loss="sparse_categorical_crossentropy", optimizer=opt, metric=["accuracy"]
    )
    return model
