import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt
from mixup import mix_up
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import tensorflow as tf
import matplotlib.pyplot as plt

# Define the mix_up function and sample_beta_distribution

# Function to create a dataset using ImageDataGenerator
def create_dataset(directory, batch_size, img_size=(224, 224), shuffle=True):
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    dataset = datagen.flow_from_directory(
        directory,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',  # Assuming categorical labels
        shuffle=shuffle
    )

    return dataset

# Function to plot images from a batch
def plot_images(images, labels, num_images=5):
    plt.figure(figsize=(15, 3))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(images[i])
        plt.title(f"Label: {tf.argmax(labels[i]).numpy()}")
        plt.axis('off')
    plt.show()

# Sample usage
data_dir = '/home/gherodes/projects/tf_test/MIT_split/test'  # Replace with the actual path to your dataset
batch_size = 32
img_size = (224, 224)

# Create two separate datasets
dataset_one = create_dataset(data_dir, batch_size, img_size)
dataset_two = create_dataset(data_dir, batch_size, img_size)

# Apply mix_up to create a new dataset
mixed_dataset = mix_up((next(iter(dataset_one)), next(iter(dataset_one))),
                       (next(iter(dataset_two)), next(iter(dataset_two))))

# Now, `mixed_dataset` contains mixed-up images and labels
# You can use it in your training loop
images, labels = mixed_dataset
print("Mixed Batch Shape:", images.shape)
print("Mixed Batch Labels Shape:", labels.shape)

# Display some images from the mixed dataset
plot_images(images, labels)
