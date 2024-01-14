import os
import getpass


from utils import *
from MLP import build_mlp
from long_residuals_mlp import build_long_residuals_mlp
from short_residuals_mlp import build_short_residuals_mlp


from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Flatten, Dense, Reshape, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model
from tensorflow.keras.initializers import HeNormal, HeUniform
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

import tensorflow as tf

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    print("Name:", gpu.name, "  Type:", gpu.device_type)

# PATHS
DATASET_DIR = '/home/gherodes/projects/tf_test/MIT_split'
MODEL_FNAME = '/home/gherodes/projects/tf_test/my_first_mlp.h5'

# HYPERPARAMS
EPOCHS = 100
PATIENCE = 10
BATCH_SIZE  = 64
LEARNING_RATE = 0.00001

MODEL_TYPE = ['no_residuals', 'short_residuals', 'long_residuals']
NUM_UNITS = [512, 1024, 2048, 4096]
IMG_SIZE    = [32,64,128.256]
IMG_SIZE = 64
REGULARIZATION_COEFF = [0.01, 0.001, 0.0001, 0.00001]
REGULARIZATION_COEFF = 0.0001


if not os.path.exists(DATASET_DIR):
  print('ERROR: dataset directory '+DATASET_DIR+' does not exist!\n')
  quit()

print('Building MLP model...\n')
# model = build_mlp(IMG_SIZE=IMG_SIZE,
#                   activation='relu')

model =  build_residual_mlp(input_shape=(IMG_SIZE, IMG_SIZE, 3),
                            activation='relu',
                            regularization_coeff=REGULARIZATION_COEFF)

lr_schedule = tf.keras.experimental.CosineDecay(
    LEARNING_RATE, decay_steps=0.75 * EPOCHS*29, alpha=0.1)

learning_rates = [lr_schedule(step).numpy() for step in range(EPOCHS*29)]
# Plot the learning rates
plt.plot(range(EPOCHS*29), learning_rates, label='Learning Rate')
plt.xlabel('Training Step')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedule')
plt.legend()
plt.savefig('learning_rate.jpg')
plt.close()

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

# initialize weights for the entire model using He normal distribution
model.kernel_initializer = HeNormal(seed=42)

print('Done!\n')

if os.path.exists(MODEL_FNAME):
  print('WARNING: model file '+MODEL_FNAME+' exists and will be overwritten!\n')

print('Create ImageDataGenerators for train and test sets...\n')

# this is the dataset configuration we will use for training
# only rescaling
train_datagen = ImageDataGenerator(
        rotation_range=45,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=False,
        brightness_range=[0.5, 1.5],
        channel_shift_range=0.2,
        rescale=1./255)

# this is the dataset configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(
   rescale=1./255)

# this is a generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
        DATASET_DIR+'/train',  # this is the target directory
        target_size=(IMG_SIZE, IMG_SIZE),  # all images will be resized to IMG_SIZExIMG_SIZE
        batch_size=BATCH_SIZE,
        classes = ['coast','forest','highway','inside_city','mountain','Opencountry','street','tallbuilding'],
        class_mode='categorical',# since we use binary_crossentropy loss, we need categorical labels
        shuffle=True)  

# this is a similar generator, for validation data
validation_generator = test_datagen.flow_from_directory(
        DATASET_DIR+'/test',
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        classes = ['coast','forest','highway','inside_city','mountain','Opencountry','street','tallbuilding'],
        class_mode='categorical')

# Define the ModelCheckpoint and EarlyStopping callbacks
checkpoint_path = MODEL_FNAME
checkpoint = ModelCheckpoint(
    checkpoint_path,
    monitor='val_accuracy',  # You can change this to a different metric if needed
    mode='max',
    save_best_only=True,
    verbose=0
)

early_stopping = EarlyStopping(
    monitor='val_accuracy',  # You can change this to a different metric if needed
    mode='max',
    patience=PATIENCE,  # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True,
    verbose=1
)
print('Fitting model')

history = model.fit(
        train_generator,
        steps_per_epoch=1881 // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=807 // BATCH_SIZE,
        callbacks=[checkpoint, early_stopping],
        verbose=1)

print('Done!\n')

print('Saving the model into '+MODEL_FNAME+' \n')
model.save_weights(MODEL_FNAME)  # always save your weights after training or during training
print('Done!\n')

  # summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('accuracy.jpg')
plt.close()
  # summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('loss.jpg')
plt.close()


#to get the output of a given layer
 #crop the model up to a certain layer
model_layer = Model(inputs=model.input, outputs=model.get_layer('second').output)

#get the features from images
directory = DATASET_DIR+'/test/coast'
x = np.asarray(Image.open(os.path.join(directory, os.listdir(directory)[0] )))
x = np.expand_dims(np.resize(x, (IMG_SIZE, IMG_SIZE, 3)), axis=0)
print('prediction for image ' + os.path.join(directory, os.listdir(directory)[0] ))
features = model_layer.predict(x/255.0)
print(features)
print('Done!')
