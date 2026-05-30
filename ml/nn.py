"""nn.py: Neural network model builder for car price regression."""

__author__ = 'Majd Jamal'

from tensorflow import keras
from tensorflow.keras import layers


def BuildNN(
    input_dim: int,
    hidden_layers: list[int],
    learning_rate: float,
    dropout: float,
    activation: str,
) -> keras.Model:
    """Builds and compiles a feed-forward neural network for regression.
    :param input_dim: Number of input features after preprocessing
    :param hidden_layers: List of hidden layer sizes
    :param learning_rate: Adam optimizer learning rate
    :param dropout: Dropout rate between hidden layers
    :param activation: Hidden layer activation function
    :return model: Compiled Keras model
    """

    inputs = keras.Input(shape=(input_dim,), name='features')
    x = inputs

    for size in hidden_layers:
        x = layers.Dense(size, activation=activation)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Dropout(dropout)(x)

    output = layers.Dense(1, name='price')(x)

    model = keras.Model(inputs=inputs, outputs=output)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae', 'mape'],
    )

    return model
