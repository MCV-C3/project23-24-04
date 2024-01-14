import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import roc_auc_score
from sklearn.metrics import RocCurveDisplay
import numpy as np

from utils.utils import *
from utils.long_residuals_mlp import *


from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Flatten, Dense, Reshape
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model
from tensorflow.keras.initializers import HeNormal, HeUniform

from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array


img_size   = 32
reg_coeff = 1e-5
num_units = 4096
filename = 'mlem-mlem-mlem'
VAL_DIR      = '/home/georg/projects/university/C3_ML_for_CV/MIT_split/test'
MODEL_FNAME  = '/home/georg/projects/university/C3_ML_for_CV/project23-24-04/Task2/long_residuals-4096-32-1e-05/long_residuals-4096-32-1e-05_best.h5'
CLASSES      = ['coast','forest','highway','inside_city','mountain','Opencountry','street','tallbuilding']

model = build_long_residuals_mlp(input_shape=(img_size, img_size, 3), 
                                         activation='relu', 
                                         regularization_coeff=reg_coeff, 
                                         num_units=num_units, 
                                         filename=f'model_{filename}.png')

model.load_weights(MODEL_FNAME)

preds = []
pred_scores = []
targets = []

for cls in os.listdir(VAL_DIR):
    subdirectory_path = os.path.join(VAL_DIR, cls)
    image_paths = [os.path.join(subdirectory_path, image_file) for image_file in os.listdir(subdirectory_path)]
    for image_path in image_paths:
        img = load_and_preprocess_image(image_path=image_path,
                                        target_size=(img_size, img_size))
 
        pred = list(model.predict(img)[0])
        pred_score = max(pred)

        pred = pred.index(max(pred))
        pred = CLASSES[pred]

        preds.append(pred)
        pred_scores.append(pred_score)
        targets.append(cls)


# plot preds


create_confusion_matrix(targets, preds)
metrics = get_metrics(targets, preds)
print(metrics)
#create and save auroc curve
RocCurveDisplay.from_predictions( targets, pred_scores)
plt.savefig('auroc.png')




