from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Flatten, Dense, Reshape, Dropout
from tensorflow.keras.utils import plot_model


def build_mlp(IMG_SIZE, num_units, activation='relu', regularization_coeff=0.01, filename='myMLP.png'):
   
  #Build the Multi Layer Perceptron model
  model = Sequential()
  model.add(Reshape((IMG_SIZE*IMG_SIZE*3,),input_shape=(IMG_SIZE, IMG_SIZE, 3),name='first'))
  model.add(Dense(units=num_units, activation=activation,name='second', kernel_regularizer=regularizers.l2(regularization_coeff)))
  model.add(Dense(units=num_units//4, activation=activation, name='third', kernel_regularizer=regularizers.l2(regularization_coeff)))
  model.add(Dense(units=num_units//16, activation=activation, name='fourth',kernel_regularizer=regularizers.l2(regularization_coeff)))
  model.add(Dense(units=8, activation='softmax'))

  print(model.summary())
  plot_model(model, to_file=filename, show_shapes=True, show_layer_names=True)

  return model