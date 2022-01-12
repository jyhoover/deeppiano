import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, RepeatVector
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical


def learning_model(Tx, LSTM_cell, densor, reshaper):
    """
    Implement the model of Tx LSTM cells where each cell is responsible
    for learning the following note based on the previous note and context.
    Each cell has the following schema:
            [X_{t}, a_{t-1}, c0_{t-1}] -> RESHAPE() -> LSTM() -> DENSE()

    Arguments:
    ==========
    - Tx:
        length of the sequences in the corpus
    - LSTM_cell:
        LSTM layer instance
    - densor:
        Dense layer instance
    - reshaper:
        Reshape layer instance

    Returns:
    ========
    - model:
        a keras instance model with inputs [X, a0, c0]
    """
    n_values = densor.units
    n_a = LSTM_cell.units

    X = Input(shape=(Tx, n_values))

    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')
    a = a0
    c = c0

    outputs = []

    for t in range(Tx):
        x = X[:, t, :]
        x = reshaper(x)
        a, _, c = LSTM_cell(inputs=x, initial_state=[a, c])
        out = densor(inputs=a)
        outputs.append(out)

    model = Model(inputs=[X, a0, c0], outputs=outputs)

    return model


def music_inference_model(LSTM_cell, densor, Ty=50):
    """
    Uses the trained "LSTM_cell" and "densor" from learning_model() to generate
    a sequence of values.

    Arguments:
    ==========
    - LSTM_cell:
        The trained "LSTM_cell" from learning_model(), Keras layer object
    - densor:
        The trained "densor" from learning_model(), Keras layer object
    - Ty:
        Integer, number of time steps to generate

    Returns:
    ========
    - inference_model:
        Keras model instance
    """

    n_values = densor.units
    n_a = LSTM_cell.units

    x0 = Input(shape=(1, n_values))

    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')
    a = a0
    c = c0
    x = x0

    outputs = []

    for t in range(Ty):
        a, _, c = LSTM_cell(inputs=x, initial_state=[a, c])
        out = densor(a)
        outputs.append(out)
        x = tf.math.argmax(out, axis=-1)
        x = tf.one_hot(indices=x, depth=n_values)
        x = RepeatVector(1)(x)

    inference_model = Model(inputs=[x0, a0, c0], outputs=outputs)

    return inference_model


def predict_and_sample(inference_model, x_initializer, a_initializer, c_initializer):
    """
    Predicts the next value of values using the inference model.

    Arguments:
    ==========
    - inference_model:
        Keras model instance for inference time
    - x_initializer:
        numpy array of shape (1, 1, n_values), one-hot vector initializing
        the values generation
    - a_initializer:
        numpy array of shape (1, n_a), initializing the hidden state
        of the LSTM_cell
    - c_initializer:
        numpy array of shape (1, n_a), initializing the cell state
        of the LSTM_cel

    Returns:
    ========
    - results:
        numpy-array of shape (Ty, n_values), matrix of one-hot vectors 
        representing the values generated
    - indices:
        numpy-array of shape (Ty, 1), matrix of indices representing
        the values generated
    """

    n_values = x_initializer.shape[2]
    pred = inference_model.predict(
        [x_initializer, a_initializer, c_initializer])
    indices = np.argmax(pred, axis=-1)
    results = to_categorical(indices, num_classes=n_values)

    return results, indices
