import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    print("Name:", gpu.name, "  Type:", gpu.device_type)

gpu_index = 0

# Set the visible devices to only the specified GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_visible_devices(gpus[gpu_index], 'GPU')

import os
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input
import keras
from keras.layers import Dense
from keras.utils import plot_model
from keras.optimizers import SGD
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils.model import CustomInceptionResNetV2 
from utils.utils import plot_learning_rate, plot_loss_and_accuracy
from utils.data_generators import create_data_generator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import optuna
import pandas as pd

DATASET_DIR = '/home/gherodes/projects/tf_test/MIT_split'
NUM_CLASSES = 8
IMG_SIZE = 256
BATCH_SIZE = 128

EPOCHS = 100
LEARNING_RATE = 0.001
PATIENCE = 10

DROPOUT = [0.0, 0.25, 0.5, 0.75, 0.9]
REG_COEFF = [0.1, 0.01, 0.001, 0.0001, 0.00001]
NUM_UNITS = [128, 256, 512]
LABEL_SMOOTH_EPSILON = [0.01, 0.05, 0.1, 0.2]
AUGMENTATIONS = [True, False]

EXPERIMENTS_DIR = './experiments/'
columns = ['Dropout', 'NumUnits', 'RegCoeff', 'LabSmooth','Augmentations', 'TrainingAccuracy', 'TrainingLoss', 'ValAccuracy', 'ValLoss']
df = pd.DataFrame([], columns=columns)

# if not os.path.isdir(EXPERIMENT_DIRECTORY):
#         os.makedirs(EXPERIMENT_DIRECTORY)
def objective(trial):
    dropout = trial.suggest_categorical('dropout', DROPOUT)
    num_units = trial.suggest_categorical('num_units', NUM_UNITS)
    reg_coeff = trial.suggest_categorical('reg_coeff', REG_COEFF)
    lab_smooth = trial.suggest_categorical('lab_smooth', LABEL_SMOOTH_EPSILON)
    augmentations = trial.suggest_categorical('augmentations', AUGMENTATIONS)


    filename = f'InceptionResnetV2-{num_units}-{dropout}-{reg_coeff}-{lab_smooth}-{augmentations}'
    experiment_directory = os.path.join(EXPERIMENTS_DIR, filename)

    if not os.path.isdir(experiment_directory):
        os.makedirs(experiment_directory)



    model = CustomInceptionResNetV2(num_classes=NUM_CLASSES,
                                    img_size=IMG_SIZE, 
                                    num_units=num_units,
                                    dropout=dropout,
                                    reg_coeff=reg_coeff,
                                    label_smooth_epsilon=lab_smooth)


    train_generator = create_data_generator(directory=DATASET_DIR + '/train',
                                            batch_size=BATCH_SIZE,
                                            img_size=IMG_SIZE,
                                            shuffle=True,
                                            augment=augmentations)
    test_generator = create_data_generator(directory=DATASET_DIR + '/test',
                                            batch_size=BATCH_SIZE,
                                            img_size=IMG_SIZE,
                                            shuffle=False,
                                            augment=False)



    num_train_files = train_generator.samples


    lr_schedule = tf.keras.experimental.CosineDecay(
        LEARNING_RATE, decay_steps=0.75 * EPOCHS*num_train_files//BATCH_SIZE, alpha=0.1)


    plot_learning_rate(lr_schedule=lr_schedule,
                    epochs=EPOCHS,
                    batches_per_epoch=num_train_files//BATCH_SIZE,
                    experiment_directory=experiment_directory)

    early_stopping = EarlyStopping(monitor='val_accuracy',  # You can change this to a different metric if needed
                                    mode='max',
                                    patience=PATIENCE,  # Number of epochs with no improvement after which training will be stopped
                                    restore_best_weights=True,
                                    verbose=0
                                            )

    checkpointer = ModelCheckpoint(experiment_directory + '/best.h5', 
                                    save_best_only=True)

    callbacks = [early_stopping, checkpointer]  # Set callbacks to None if you don't have any custom callbacks

    history = model.train(train_generator, test_generator, epochs=100, callbacks=callbacks)

    plot_loss_and_accuracy(history, experiment_directory)

    model.save_model(f'{experiment_directory}/last.h5')

    config = [dropout, num_units, reg_coeff, lab_smooth, augmentations, history.history['accuracy'], history.history['loss'], history.history['val_accuracy'], history.history['val_loss']]
  
    df.loc[len(df)] = config
    df.reset_index(drop=True, inplace=True)
    print( max(history.history['val_accuracy']))
    return max(history.history['val_accuracy'])

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
df.to_csv(f'{EXPERIMENTS_DIR}/logs.csv')
study.best_params

