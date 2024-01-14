#from __future__ import print_function
import os,sys
import numpy as np
from sklearn.feature_extraction import image
from tensorflow.keras.preprocessing import image as tf_image
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import os 
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import pandas as pd
from numpy import genfromtxt

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def generate_image_patches_db(in_directory,out_directory,patch_size=64):
  if not os.path.exists(out_directory):
      os.makedirs(out_directory)
 
  total = 2688
  count = 0  
  for split_dir in os.listdir(in_directory):
    if not os.path.exists(os.path.join(out_directory,split_dir)):
      os.makedirs(os.path.join(out_directory,split_dir))
  
    for class_dir in os.listdir(os.path.join(in_directory,split_dir)):
      if not os.path.exists(os.path.join(out_directory,split_dir,class_dir)):
        os.makedirs(os.path.join(out_directory,split_dir,class_dir))
  
      for imname in os.listdir(os.path.join(in_directory,split_dir,class_dir)):
        count += 1
        im = Image.open(os.path.join(in_directory,split_dir,class_dir,imname))
        print('Processed images: '+str(count)+' / '+str(total), end='\r')
        patches = image.extract_patches_2d(np.array(im), (64, 64), max_patches=1)
        for i,patch in enumerate(patches):
          patch = Image.fromarray(patch)
          patch.save(os.path.join(out_directory,split_dir,class_dir,imname.split(',')[0]+'_'+str(i)+'.jpg'))
  print('\n')


def load_and_preprocess_image(image_path, target_size=(32, 32)):
    """
    Preprocesses an image for inference.

    Args:
        image_path (str): Path to the image file.
        target_size (tuple): Target size for resizing the image.

    Returns:
        numpy.ndarray: Preprocessed image as a NumPy array.
    """
    # Load the image
    img = tf_image.load_img(image_path, target_size=target_size)
    # Convert the image to a NumPy array
    img_array = img_to_array(img)
    # Reshape the image to have a batch size of 1 (optional, depending on your model)
    img_array = np.expand_dims(np.resize(img_array, (target_size[0], target_size[1], 3)), axis=0)
    # Normalize the pixel values (assuming you used this normalization during training)
    img_array /= 255.0
    return img_array



def create_confusion_matrix(targets, preds, classes=None):

    if classes == None:
        classes = list(set(targets))
    cf_matrix = confusion_matrix(targets, preds, normalize='true')
    df_cm = pd.DataFrame(cf_matrix, index = [i for i in classes],
                    columns = [i for i in classes])
    
    plt.figure(figsize = (12,7))       
    sn.heatmap(df_cm, annot=True)
    plt.savefig('confusion_matrix.png')
    plt.close()     

def get_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred, average='macro')
    precision = precision_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
 
    return accuracy, recall, precision, f1   

def load_data_from_directory(path):
    image_paths = []
    labels = []
    for class_directory in os.listdir(path):
        image_file_names = os.listdir(os.path.join(path, class_directory))
        for image_file_name in image_file_names:
            image_paths.append(os.path.join(path, class_directory, image_file_name))
            labels.append(class_directory)
    return image_paths, labels

def read_embeddings(directory):
  embeddings = {}
  for embedding_file_name in os.listdir(directory):
    embedding_path = os.path.join(directory, embedding_file_name)
    embedding = genfromtxt(embedding_path, delimiter=',')
    embeddings[embedding_file_name.replace('csv', 'jpg')] = embedding
  return embeddings




# function to split 1x265x265x3 image into patches of specific size
def split_image_into_patches(image, patch_size):
    patches = []
    
    for i in range(0, image.shape[1], patch_size):
        for j in range(0, image.shape[2], patch_size):
            patch = image[:, i:i+patch_size, j:j+patch_size, :]
            patches.append(patch)
    return np.array(patches)

