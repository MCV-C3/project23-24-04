from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

import cv2
import os
import torch
import matplotlib.pyplot as plt

from utils.model import MITClassifier
from utils.transforms import preprocess

directory = '/home/gherodes/projects/tf_test/dataset/MIT_small_train_1/test/'
CLASSES = ['coast','forest','highway','inside_city','mountain','Opencountry','street','tallbuilding']
NUM_CLASSES = len(CLASSES)
WEIGHTS_PATH = '/home/gherodes/projects/tf_test/project23-24-04/Task4/lightning_logs/version_74/checkpoints/epoch=29-step=240.ckpt'

VIZ_PATH = '/home/gherodes/projects/tf_test/project23-24-04/Task4/gradcams'


model = MITClassifier.load_from_checkpoint(WEIGHTS_PATH, 
                                           num_classes=NUM_CLASSES,
                                           class_names=CLASSES)
model.eval()

target_layers = model.feature_extractor[-2]
# Construct the CAM object once, and then re-use it on many images:


for class_ in CLASSES:
    input_list = []
    imgs = []
    targets = []
    input_tensor = None
    cam = EigenCAM(model=model, target_layers=target_layers)

    class_path = os.path.join(directory, class_)
    files = [f for f in os.listdir(class_path) if f.endswith(('jpg', 'png'))]
    if len(files) > 16:
        files = files[:16]
    for file in files:  
        file_path = os.path.join(class_path, file)
        img = cv2.imread(file_path)

        ref_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ref_img_resized = cv2.resize(ref_img, [256,256])/255

        img = preprocess(dims=(256,256))(image=img)['image']#.unsqueeze(0)
        imgs.append(ref_img_resized)
        input_list.append(img)
        targets.append(ClassifierOutputTarget(int(class_)-1))

    input_tensor = torch.stack(input_list)
    # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)
    preds = cam.outputs

    visualizations = []
    for i, gs_cam in enumerate(grayscale_cam):
        img_name = files[i]
        visualization = show_cam_on_image(imgs[i], gs_cam, use_rgb=True)
        visualizations.append(visualization)


    # Create a 4x4 grid    
    fig, axs = plt.subplots(4, 4, figsize=(20, 20))

    for i, (image) in enumerate(visualizations):
        # Calculate the row and column for the subplot
        row, col = divmod(i, 4)
        axs[row, col].imshow(image)
        axs[row, col].axis('off')
        pred = CLASSES[int(preds[i].argmax(0))]
        axs[row, col].set_title(f'Predicted: {pred}', fontsize=15)

    plt.suptitle(f'Class: {class_}', fontsize=36)
    plt.savefig(f'./gradcams/gradcam_{class_}.png')
    plt.close()


    # You can also get the model outputs without having to re-inference
    # model_outputs = cam.outputs
