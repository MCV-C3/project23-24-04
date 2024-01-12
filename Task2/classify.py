import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

from utils import *
from MLP import *


from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Flatten, Dense, Reshape
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model
from tensorflow.keras.initializers import HeNormal, HeUniform

from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array


IMG_SIZE     = 64
VAL_DIR      = '/home/gherodes/projects/tf_test/MIT_split/test'
MODEL_FNAME  = '/home/gherodes/projects/tf_test/my_first_mlp.h5'
CLASSES      = ['coast','forest','highway','inside_city','mountain','Opencountry','street','tallbuilding']

model = build_mlp(IMG_SIZE=IMG_SIZE,
                  activation='relu')

model.load_weights(MODEL_FNAME)

preds = []
targets = []

model_layer = Model(inputs=model.input, outputs=model.get_layer('fifth').output)

for cls in os.listdir(VAL_DIR):
    subdirectory_path = os.path.join(VAL_DIR, cls)
    image_paths = [os.path.join(subdirectory_path, image_file) for image_file in os.listdir(subdirectory_path)]
    for image_path in image_paths:
        img = load_and_preprocess_image(image_path=image_path,
                                        target_size=(IMG_SIZE, IMG_SIZE))
 
        pred = list(model.predict(img)[0])
        pred = pred.index(max(pred))
        pred = CLASSES[pred]

        preds.append(pred)
        targets.append(cls)

        layer_output = model_layer.predict(img)
        print(layer_output)


create_confusion_matrix(targets, preds)
metrics = get_metrics(targets, preds)
print(metrics)


