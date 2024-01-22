import numpy as np
import tensorflow as tf
import keras

import os
os.environ["KERAS_BACKEND"] = "tensorflow"

# TF imports related to tf.data preprocessing
from tensorflow import data as tf_data
from tensorflow import image as tf_image
from tensorflow.random import gamma as tf_random_gamma


import matplotlib.pyplot as plt

from keras import layers

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator



# Function to sample from Beta distribution
def sample_beta_distribution(size, concentration_0=0.2, concentration_1=0.2):
    gamma_1_sample = tf.random.gamma(shape=[size], alpha=concentration_1)
    gamma_2_sample = tf.random.gamma(shape=[size], alpha=concentration_0)
    return gamma_1_sample / (gamma_1_sample + gamma_2_sample)

# Function to apply MixUp
def mixup(data, alpha=0.2):
    print('DOING MIXUP')
    images, labels = data
    batch_size = tf.shape(images)[0]

    # Sample lambda from Beta distribution
    l = sample_beta_distribution(batch_size, alpha, alpha)
    x_l = tf.reshape(l, (batch_size, 1, 1, 1))
    y_l = tf.reshape(l, (batch_size, 1))

    # Perform MixUp on both images and labels
    mixed_images = images * x_l + tf.reverse(images, axis=[0]) * (1 - x_l)
    mixed_labels = labels * y_l + tf.reverse(labels, axis=[0]) * (1 - y_l)

    return mixed_images, mixed_labels
# Function to apply CutMix
def cutmix(data, alpha=0.2):
    images, labels = data
    batch_size = tf.shape(images)[0]

    # Sample lambda from Beta distribution
    l = sample_beta_distribution(batch_size, alpha, alpha)
    x_l = tf.reshape(l, (batch_size, 1, 1, 1))
    y_l = tf.reshape(l, (batch_size, 1))

    # Randomly choose another image and its label from the batch
    shuffled_indices = tf.range(batch_size)
    shuffled_indices = tf.random.shuffle(shuffled_indices)
    
    mixed_images = images * x_l + tf.gather(images, shuffled_indices) * (1 - x_l)
    mixed_labels = labels * y_l + tf.gather(labels, shuffled_indices) * (1 - y_l)

    return mixed_images, mixed_labels

if __name__ == '__main__':

    # Define the directory path where your dataset is stored
    dataset_directory = '/home/gherodes/projects/tf_test/MIT_split/train'

    # Create an ImageDataGenerator for data augmentation
    datagen = ImageDataGenerator(
        rescale=1./255,  # Normalize pixel values to [0, 1]
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    # Flow images from the directory using the ImageDataGenerator
    # Specify the target size, batch size, and class mode as needed
    batch_size = 32
    image_size = (256, 256)  # Adjust according to your dataset
    class_mode = 'categorical'  # Use 'binary' for binary classification

    train_generator = datagen.flow_from_directory(
        directory=dataset_directory,
        target_size=image_size,
        batch_size=batch_size,
        class_mode=class_mode,
        shuffle=True  # Set to False if you want to keep the order
    )


    train_dataset = tf.data.Dataset.from_generator(
        lambda: train_generator,
        output_signature=(
            tf.TensorSpec(shape=(batch_size, image_size[0], image_size[1], 3), dtype=tf.float32),
            tf.TensorSpec(shape=(batch_size, 8), dtype=tf.float32)
        )
    )

    # Apply MixUp and CutMix using the map function
    alpha_mixup = 0.2
    alpha_cutmix = 0.2


    # Apply MixUp using the map function
    mixed_generator = train_dataset.map(lambda x, y: mixup((x, y), alpha=alpha_mixup))
    cutmixed_generator = train_dataset.map(lambda x, y: cutmix((x, y), alpha=alpha_cutmix))

    # Function to plot images
    def plot_images(images, labels, title, savename):
        plt.figure(figsize=(20, 20))
        for i in range(min(9, images.shape[0])):
            plt.subplot(3, 3, i + 1)
            plt.imshow(images[i])
            plt.title(f"Label: {labels[i]}")
            plt.axis('off')
        plt.suptitle(title)
        plt.savefig(savename)

    # Display images from mixed_generator
    for batch in mixed_generator:
        mixed_images, mixed_labels = batch
        plot_images(mixed_images.numpy(), mixed_labels.numpy(), title="Mixed Images", savename='mixup.jpg')
        break  # Display only the first batch

    # Display images from cutmixed_generator
    for batch in cutmixed_generator:
        cutmixed_images, cutmixed_labels = batch
        plot_images(cutmixed_images.numpy(), cutmixed_labels.numpy(), title="CutMixed Images",savename='cutmix.jpg')
        break  # Display only the first batch
