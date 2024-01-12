from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Flatten, Dense, Reshape, Dropout
from tensorflow.keras.utils import plot_model


def build_mlp(IMG_SIZE, activation='relu'):
   
  #Build the Multi Layer Perceptron model
  model = Sequential()
  model.add(Reshape((IMG_SIZE*IMG_SIZE*3,),input_shape=(IMG_SIZE, IMG_SIZE, 3),name='first'))
  model.add(Dense(units=4096, activation=activation,name='second'))
  model.add(Dense(units=4096, activation=activation, name='third'))
  model.add(Dense(units=1024, activation=activation, name='fourth'))
  model.add(Dense(units=8, activation='softmax'))

  print(model.summary())
  plot_model(model, to_file='modelMLP.png', show_shapes=True, show_layer_names=True)

  return model