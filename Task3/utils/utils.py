import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def plot_learning_rate(lr_schedule, epochs, batches_per_epoch, experiment_directory):
    learning_rates = [lr_schedule(step).numpy() for step in range(epochs*batches_per_epoch)]
    # Plot the learning rates
    plt.plot(range(epochs*batches_per_epoch), learning_rates, label='Learning Rate')
    plt.xlabel('Training Step')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.legend()
    plt.savefig(f'{experiment_directory}/learning_rate.jpg')
    plt.close()

def plot_loss_and_accuracy(history, experiment_directory):
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