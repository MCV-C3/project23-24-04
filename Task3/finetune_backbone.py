import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import tensorflow as tf
import keras
#from IPython.display import Image, display
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from utils.model import CustomInceptionResNetV2 
from utils.utils import plot_learning_rate, plot_loss_and_accuracy
from utils.data_generators import create_data_generator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from utils.mixup_cutmix import mixup, cutmix
from tensorflow.keras import layers, models, regularizers

from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input

from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.models import Model
from utils.label_smoothing import label_smooth

   
DATASET = '/home/gherodes/projects/tf_test/dataset/MIT_split/'
BATCH_SIZE = 64
IMG_SIZE = 256
LEARNING_RATE = 0.001
EPOCHS = 100
PATIENCE = 10
EXPERIMENT_DIRECTORY = 'finetune'
LABEL_SMOOTH_EPSILON=0.1
REG_COEFF = 0.01

if not os.path.isdir(EXPERIMENT_DIRECTORY):
        os.makedirs(EXPERIMENT_DIRECTORY)


#hyperparams
NUM_BLOCKS_TO_RETRAIN = [0,1,2,3,4]
block_end_names = ['block8_9_ac',  'block8_8_ac', 'block8_7_ac', 'block8_6_ac', 'block8_5_ac']
block_end_indices = [761, 745, 729, 713, 697]


# create the base pre-trained model
base_model = InceptionResNetV2(weights='imagenet', include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='gelu',kernel_regularizer=regularizers.l2(REG_COEFF))(x)
x = Dropout(0.5)(x)
predictions = Dense(8, activation='softmax')(x)

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)
model.load_weights('/home/gherodes/projects/tf_test/project23-24-04/Task3/experiments/InceptionResnetV2-256-0.5-0.001-0.01-True/best.h5', by_name=True)

# we chose to train the top 2 inception blocks, i.e. we will freeze
# the first 249 layers and unfreeze the rest:
for layer in model.layers[:761]:
   layer.trainable = False
for layer in model.layers[761:]:
   layer.trainable = True
train_generator = create_data_generator(directory=DATASET+ '/train',
                                        batch_size=BATCH_SIZE,
                                        img_size=IMG_SIZE,
                                        shuffle=True,
                                        augment=True)
num_train_files = train_generator.samples
test_generator = create_data_generator(directory=DATASET + '/test',
                                        batch_size=BATCH_SIZE,
                                        img_size=IMG_SIZE,
                                        shuffle=False,
                                        augment=False)


lr_schedule = tf.keras.experimental.CosineDecay(
    LEARNING_RATE, decay_steps=0.75 * EPOCHS*num_train_files//BATCH_SIZE, alpha=0.1)


plot_learning_rate(lr_schedule=lr_schedule,
                   epochs=EPOCHS,
                   batches_per_epoch=num_train_files//BATCH_SIZE,
                   experiment_directory=EXPERIMENT_DIRECTORY)

early_stopping = EarlyStopping(monitor='val_accuracy',  # You can change this to a different metric if needed
                                mode='max',
                                patience=PATIENCE,  # Number of epochs with no improvement after which training will be stopped
                                restore_best_weights=True,
                                verbose=0
                                        )

checkpointer = ModelCheckpoint(EXPERIMENT_DIRECTORY + '/best.h5', 
                                save_best_only=True)

callbacks = [early_stopping, checkpointer]

if LABEL_SMOOTH_EPSILON > 0:
    loss=lambda y_true, y_pred: tf.keras.losses.categorical_crossentropy(label_smooth(y_true, LABEL_SMOOTH_EPSILON), y_pred)
else: 
    loss = 'categorical_crossentropy'
        
model.compile(optimizer='adam',
                    loss=loss,
                    metrics=['accuracy'])  # Set callbacks to None if you don't have any custom callbacks

history = model.fit(
            train_generator,
            epochs=EPOCHS,
            validation_data=test_generator,
            callbacks=callbacks,
            verbose=1
        )

plot_loss_and_accuracy(history, EXPERIMENT_DIRECTORY)

model.save_model(f'{EXPERIMENT_DIRECTORY}/last.h5')


