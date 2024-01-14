import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.utils import plot_model


def network(x,  activation, regularization_coeff, num_units=128):
    """Residual block with skip connection using concatenation."""
    shortcut_1 = x
    x = layers.Dense(num_units, activation=activation, kernel_regularizer=regularizers.l2(regularization_coeff))(x)
    x = layers.Dense(num_units, activation=None, kernel_regularizer=regularizers.l2(regularization_coeff))(x)
    shortcut_2 = x
    x = layers.Dense(num_units//4, activation=activation, kernel_regularizer=regularizers.l2(regularization_coeff))(x)
    x = layers.Dense(num_units//4, activation=None, kernel_regularizer=regularizers.l2(regularization_coeff))(x)
    shortcut_3 = x
    x = layers.Dense(num_units//16, activation=activation, kernel_regularizer=regularizers.l2(regularization_coeff))(x)
    x = layers.Dense(num_units//16, activation=None, kernel_regularizer=regularizers.l2(regularization_coeff))(x)
   
    x = layers.Concatenate()([x, shortcut_3])

    x = layers.Dense(num_units//4, activation=activation, kernel_regularizer=regularizers.l2(regularization_coeff))(x)
    x = layers.Dense(num_units//4, activation=None, kernel_regularizer=regularizers.l2(regularization_coeff))(x)
    x = layers.Concatenate()([x, shortcut_2])
    
    x = layers.Dense(num_units, activation=activation, kernel_regularizer=regularizers.l2(regularization_coeff))(x)
    x = layers.Dense(num_units, activation=None, kernel_regularizer=regularizers.l2(regularization_coeff))(x)
    x = layers.Concatenate()([x, shortcut_1])

    x = layers.Activation(activation)(x)
    return x

def build_long_residuals_mlp(input_shape, activation='relu', regularization_coeff=0.01, num_units=128, filename='myMLP.png'):
    """Build a fully connected residual MLP for image classification."""
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Flatten()(inputs)

   
    x = network(x, activation=activation, regularization_coeff=regularization_coeff, num_units=num_units)

    # Fully connected layer for classification
    outputs = layers.Dense(8, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    print(model.summary())
    plot_model(model, to_file=filename, show_shapes=True, show_layer_names=True)

    return model