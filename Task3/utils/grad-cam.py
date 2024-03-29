import os
os.environ["KERAS_BACKEND"] = "tensorflow"

import numpy as np
import tensorflow as tf
import keras
#from IPython.display import Image, display
import matplotlib as mpl
import matplotlib.pyplot as plt

from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input

from keras.layers import Dense, GlobalAveragePooling2D
from keras.models import Model

from model import CustomInceptionResNetV2 


### Functions ###
def get_img_array(img_path, size):
    # `img` is a PIL image of size 299x299
    img = keras.utils.load_img(img_path, target_size=size)
    # `array` is a float32 Numpy array of shape (299, 299, 3)
    array = keras.utils.img_to_array(img)
    # We add a dimension to transform our array into a "batch"
    # of size (1, 299, 299, 3)
    array = np.expand_dims(array, axis=0)
    return array


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = keras.models.Model(
        model.inputs, 
        [model.get_layer(last_conv_layer_name).output, 
        model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()



def save_and_display_gradcam(img_path, heatmap, cam_path="cam.jpg", alpha=0.4):
    # Load the original image
    img = keras.utils.load_img(img_path)
    img = keras.utils.img_to_array(img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = mpl.colormaps["jet"]

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)

    # Superimpose the heatmap on original image
    superimposed_img = jet_heatmap * alpha + img
    superimposed_img = keras.utils.array_to_img(superimposed_img)

    # Save the superimposed image
    superimposed_img.save(cam_path)

    # # Display Grad CAM
    # display(Image(cam_path))

################



if __name__ == '__main__':
    
    NUM_CLASSES = 8
    BATCH_SIZE = 128
    DROPOUT = 0.5

    directory = '/home/gherodes/projects/tf_test/dataset/MIT_small_train_1/train'
    classes = sorted(os.listdir(directory))
    classes_files = []
    for cls in classes:
        path = os.path.join(directory, cls)
        file = os.listdir(path)[0]
        filepath = os.path.join(path, file)
        classes_files.append([cls,filepath])

    # create the base pre-trained model
    base_model = InceptionResNetV2(weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(256, activation='gelu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(8, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)
    model.load_weights('/home/gherodes/projects/tf_test/project23-24-04/Task3/experiments/small_1/best.h5', by_name=True)
    img_size = (256, 256)

    last_conv_layer_name = "conv_7b_ac"
    for cls, file in classes_files:
        # The local path to our target image
        img_path = file
        print(img_path)

        # Prepare image
        img_array = preprocess_input(get_img_array(img_path, size=img_size))

        # Remove last layer's softmax
        model.layers[-1].activation = None

        # Print what the top predicted class is
        preds = model.predict(img_array)
        print(preds.shape)
        print("Predicted:", ['coast','forest','highway','inside_city','mountain','Opencountry','street','tallbuilding'][preds[0].tolist().index(max(preds[0]))])
        print(f'Expected: {cls}')

        # Generate class activation heatmap
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name)

        # Display heatmap
        plt.matshow(heatmap)
        plt.savefig('GRADCAM.jpg')
        save_and_display_gradcam(img_path, heatmap, cam_path=f'gradcams/{cls}.jpg')









