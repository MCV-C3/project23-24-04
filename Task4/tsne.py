import cv2
import os
import torch
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm

from utils.model import MITClassifier
from utils.transforms import preprocess
from utils.data import MITDataModule
from sklearn.manifold import TSNE
from torchvision.models.feature_extraction import create_feature_extractor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

directory = DATA_ROOT = '/home/gherodes/projects/tf_test/dataset/MIT_small_train_1'


CLASSES = ['coast','forest','highway','inside_city','mountain','Opencountry','street','tallbuilding']
NUM_CLASSES = len(CLASSES)
WEIGHTS_PATH = '/home/gherodes/projects/tf_test/project23-24-04/Task4/lightning_logs/version_74/checkpoints/epoch=29-step=240.ckpt'

colors_per_class = {
    1: ((255, 0, 0)),    # Red
    2: ((0, 255, 0)),    # Green
    3: ((0, 0, 255)),    # Blue
    4: ((255, 255, 0)),  # Yellow
    5: ((255, 0, 255)),  # Magenta
    6: ((0, 255, 255)),  # Cyan
    7: ((128, 0, 0)),    # Maroon
    8: ((0, 128, 0)),    # Green (dark)
    9: ((0, 0, 128)),    # Navy
    10: ((128, 128, 0)),  # Olive
    11: ((128, 0, 128)),  # Purple
    12: ((0, 128, 128)),  # Teal
}
model = MITClassifier.load_from_checkpoint(WEIGHTS_PATH, 
                                           num_classes=NUM_CLASSES,
                                           class_names=CLASSES)

model.eval()
print(model)

LAYER_NUM = None

print(model.feature_extractor[:LAYER_NUM][-1])

dm = MITDataModule(batch_size=16, 
                   root_dir=directory,
                   dims=(256,256),
                   )
dm.setup()
dataloader = dm.val_dataloader()
outputs = []
all_labels = []
for batch in tqdm(dataloader, desc='Running the model inference'):
    
    images = batch[0].to(device)
    labels = batch[1]

    output = model.forward_intermediate(images, LAYER_NUM)
    current_outputs = output.cpu().detach().numpy()
    #current_outputs = output.cpu().numpy()
    outputs.append(current_outputs)
    all_labels.append(labels.cpu().detach().numpy())


features = np.vstack(outputs)
all_labels = np.hstack(all_labels)
print(features.shape)
tsne = TSNE(n_components=2).fit_transform(features)

# scale and move the coordinates so they fit [0; 1] range
def scale_to_01_range(x):
    # compute the distribution range
    value_range = (np.max(x) - np.min(x))
 
    # move the distribution so that it starts from zero
    # by extracting the minimal value from all its values
    starts_from_zero = x - np.min(x)
 
    # make the distribution fit [0; 1] by dividing by its range
    return starts_from_zero / value_range
 
# extract x and y coordinates representing the positions of the images on T-SNE plot
tx = tsne[:, 0]
ty = tsne[:, 1]
 
tx = scale_to_01_range(tx)
ty = scale_to_01_range(ty)
	
# initialize a matplotlib plot
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)
 
# for every class, we'll add a scatter plot separately
all_labels = [item + 1 for item in all_labels]

for label in colors_per_class:
    # find the samples of the current class in the data
    indices = [i for i, l in enumerate(all_labels) if l == label]
    
    # extract the coordinates of the points of this class only
    current_tx = np.take(tx, indices)
    current_ty = np.take(ty, indices)
 
    # convert the class color to matplotlib format
    color = np.array(colors_per_class[label], dtype=float) / 255
 
    # add a scatter plot with the corresponding color and label
    ax.scatter(current_tx, current_ty, c=color, label=label)
 
# build a legend using the labels we set previously
ax.legend(loc='best')
plt.suptitle('T-SNE Clustering of Representations ')
# finally, show the plot
plt.savefig('TSNE.jpg')


