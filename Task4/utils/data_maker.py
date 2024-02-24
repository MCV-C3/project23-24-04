from transforms import *
import os
from PIL import Image
import numpy as np

src_directory = "/ghome/group04/project23-24-04/datasets/MIT_small_train_1/train"
dest_directory = (
    "/ghome/group04/project23-24-04/datasets/MIT_small_train_1_augmented/train"
)

NUM_AUGMENTATIONS = 5

classes = os.listdir(src_directory)


def apply_augmentation(image_path, output_path, augmentations):
    # Load image using PIL
    original_image = np.array(Image.open(image_path))
    # Apply augmentations
    augmented_image = augmentations(image=original_image)

    # Save augmented image to disk
    augmented_image_path = output_path
    augmented_image = Image.fromarray(augmented_image["image"])
    augmented_image.save(augmented_image_path)


for class_ in classes:
    class_src_directory = os.path.join(src_directory, class_)
    class_dest_directory = os.path.join(dest_directory, class_)
    if not os.path.exists(class_dest_directory):
        os.makedirs(class_dest_directory)

    image_files = os.listdir(class_src_directory)
    for image_file in image_files:
        img_src_path = os.path.join(class_src_directory, image_file)

        for i in range(0, NUM_AUGMENTATIONS):
            dest_img_file = (
                image_file.split(".")[0] + f"_{i}." + image_file.split(".")[-1]
            )
            img_dest_path = os.path.join(class_dest_directory, dest_img_file)
            apply_augmentation(img_src_path, img_dest_path, albumentations_transform())
