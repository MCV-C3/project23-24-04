import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.utils import plot_model


def residual_block(x, num_units, activation, regularization_coeff):
    """Residual block with skip connection using concatenation."""
    shortcut = x
    x = layers.Dense(num_units, activation=activation, kernel_regularizer=regularizers.l2(regularization_coeff))(x)
    x = layers.Dense(num_units//4, activation=None, kernel_regularizer=regularizers.l2(regularization_coeff))(x)
    # Concatenate instead of adding
    x = layers.Concatenate()([x, shortcut])
    x = layers.Activation(activation)(x)
    return x

def build_short_residuals_mlp(input_shape, num_units, activation='relu',regularization_coeff=0.01, filename='myMPL.png'):
    """Build a fully connected residual MLP for image classification."""
    num_units_list = [num_units, num_units//4, num_units//16, num_units//64]
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Flatten()(inputs)

    # Residual blocks
    for num_units in num_units_list:
        x = residual_block(x, num_units, activation=activation, regularization_coeff=regularization_coeff)

    # Fully connected layer for classification
    outputs = layers.Dense(8, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    print(model.summary())
    plot_model(model, to_file=filename, show_shapes=True, show_layer_names=True)

    return model


