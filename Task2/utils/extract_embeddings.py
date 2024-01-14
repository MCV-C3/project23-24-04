from long_residuals_mlp import *
from utils import *
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Flatten, Dense, Reshape
import cv2


PATCH_SIZE = 32
CLASSES      = ['coast','forest','highway','inside_city','mountain','Opencountry','street','tallbuilding']

MODEL_WEIGHTS = '/home/georg/projects/university/C3_ML_for_CV/project23-24-04/Task2/long_residuals-4096-32-1e-05/long_residuals-4096-32-1e-05_best.h5'
images_directory = '/home/georg/projects/university/C3_ML_for_CV/MIT_split/'


model = build_long_residuals_mlp(input_shape=(PATCH_SIZE, PATCH_SIZE, 3),
                                  activation='relu', 
                                  regularization_coeff=0.00001, 
                                  num_units=4096, 
                                  filename='myMLP.png')

model.load_weights(MODEL_WEIGHTS)

model_layer = Model(inputs=model.input, outputs=model.get_layer('dense_5').output)



for subdir1 in os.listdir(images_directory):
    for subdir2 in os.listdir(os.path.join(images_directory, subdir1)):
        for file in os.listdir(os.path.join(images_directory, subdir1, subdir2)):
            img = load_and_preprocess_image(image_path=os.path.join(images_directory, subdir1, subdir2, file),
                                            target_size=(256, 256,3))
            

            patches = split_image_into_patches(img, PATCH_SIZE)
            print(patches.shape)
            patch_embeddings = []
            for patch in patches:
                layer_output = model_layer.predict(patch)
                patch_embeddings.append(layer_output)
            layer_output = np.array(patch_embeddings).squeeze(axis=1)
            print(layer_output.shape)

            # save np array to csv file
            np.savetxt(os.path.join('/home/georg/projects/university/C3_ML_for_CV/project23-24-04/Task2/embeddings/32', file.split('.')[0]+'.csv'), layer_output, delimiter=',')