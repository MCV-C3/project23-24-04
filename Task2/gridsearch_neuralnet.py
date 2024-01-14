import optuna

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
import pandas as pd


gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    print("Name:", gpu.name, "  Type:", gpu.device_type)

# PATHS
DATASET_DIR = '/home/gherodes/projects/tf_test/MIT_split'
EXPERIMENTS_DIR = '/home/gherodes/projects/tf_test/project23-24-04/Task2/experiments'

# HYPERPARAMS
EPOCHS = 100
PATIENCE = 10
BATCH_SIZE  = 64
LEARNING_RATE = 0.00001

MODEL_TYPE = ['no_residuals', 'short_residuals', 'long_residuals']
NUM_UNITS = [512, 1024, 2048, 4096]
IMG_SIZE    = [32,64,128]
REGULARIZATION_COEFF = [0.01, 0.001, 0.0001, 0.00001]

df = pd.DataFrame([], columns=['ModelType', 'NumUnits', 'ImgSize', 'RegCoeff', 'Accuracy', 'TrainingLoss', 'ValAccuracy', 'ValLoss'])

def objective(trial):
    model_type = trial.suggest_categorical('model_type', MODEL_TYPE)
    num_units = trial.suggest_categorical('num_units', NUM_UNITS)
    img_size = trial.suggest_categorical('img_size', IMG_SIZE)
    reg_coeff = trial.suggest_categorical('reg_coeff', REGULARIZATION_COEFF)

    print(f'Model Type: {model_type}')
    print(f'Num Units : {num_units}')
    print(f'Img Size: {img_size}')
    print(f'Reg Coeff: {reg_coeff}')


    filename = f'{model_type}-{num_units}-{img_size}-{reg_coeff}'
    experiment_directory = os.path.join(EXPERIMENTS_DIR, filename)

    if not os.path.isdir(experiment_directory):
        os.makedirs(experiment_directory)

    if model_type == 'no_residuals':
        model = build_mlp(img_size,
                          num_units,
                          activation='relu',
                          regularization_coeff=reg_coeff,
                          filename=f'model_{filename}.png'
                        )
    elif model_type == 'short_residuals':
        model = build_short_residuals_mlp(input_shape=(img_size, img_size, 3),
                                        num_units=num_units,
                                        activation='relu',
                                        regularization_coeff=reg_coeff, 
                                        filename=f'model_{filename}.png')
        
    elif model_type == 'long_residuals':
        model = build_long_residuals_mlp(input_shape=(img_size, img_size, 3), 
                                         activation='relu', 
                                         regularization_coeff=reg_coeff, 
                                         num_units=num_units, 
                                         filename=f'model_{filename}.png')
    
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
            target_size=(img_size, img_size),  # all images will be resized to IMG_SIZExIMG_SIZE
            batch_size=BATCH_SIZE,
            classes = ['coast','forest','highway','inside_city','mountain','Opencountry','street','tallbuilding'],
            class_mode='categorical',# since we use binary_crossentropy loss, we need categorical labels
            shuffle=True)  

    # this is a similar generator, for validation data
    validation_generator = test_datagen.flow_from_directory(
            DATASET_DIR+'/test',
            target_size=(img_size, img_size),
            batch_size=BATCH_SIZE,
            classes = ['coast','forest','highway','inside_city','mountain','Opencountry','street','tallbuilding'],
            class_mode='categorical')

    # Define the ModelCheckpoint and EarlyStopping callbacks
    checkpoint_path = os.path.join(experiment_directory, filename+'_best.h5')
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
        verbose=0
    )
    print('Fitting model')

    history = model.fit(
            train_generator,
            steps_per_epoch=1881 // BATCH_SIZE,
            epochs=EPOCHS,
            validation_data=validation_generator,
            validation_steps=807 // BATCH_SIZE,
            callbacks=[checkpoint, early_stopping],
            verbose=0)

    print('Done!\n')

    model.save_weights(os.path.join(experiment_directory, filename + '_final_.h5'))  # always save your weights after training or during training

    # summarize history for accuracy
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(f'{experiment_directory}/accuracy.jpg')
    plt.close()
    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(f'{experiment_directory}/loss.jpg')
    plt.close()
    config = [model_type, num_units, img_size, reg_coeff, history.history['accuracy'], history.history['loss'], history.history['val_accuracy'], history.history['val_loss']]
    df.loc[len(df)] = config
    df.reset_index(drop=True, inplace=True)
    print( max(history.history['val_accuracy']))
    return max(history.history['val_accuracy'])

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
df.to_csv(f'{EXPERIMENTS_DIR}/logs.csv')
study.best_params


