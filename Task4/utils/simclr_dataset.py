import datasets

import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.utils import make_grid

import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image
import sys

from .transforms import contrast_transforms


class SimCLRDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform=None, num_views=2):
        self.dataset = dataset
        self.transform = transform
        self.num_views = num_views

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image = Image.fromarray(np.array(self.dataset[idx]["image"]))
        label = self.dataset[idx]["label"]
        if image.mode == "L":
            image = image.convert("RGB")
        views = []
        for i in range(0, self.num_views):
            views.append(self.transform(image))

        return views, label


if __name__ == "__main__":

    dataset = datasets.load_dataset("imagenet-1k")["train"]
    dataset = SimCLRDataset(dataset, contrast_transforms)

    print(dataset)
    print("making dataloader")
    # Create a DataLoader using the custom dataset
    batch_size = 16
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        drop_last=False,
    )
    print("getting batches")

    for batch in dataloader:
        (image1s, image2s), _ = batch

        # Pair up images from image1s and image2s at corresponding indices
        paired_images = [(img1, img2) for img1, img2 in zip(image1s, image2s)]

        # Create a grid of paired images
        grid = make_grid(
            [torch.cat((img1, img2), dim=2) for img1, img2 in paired_images],
            nrow=4,
            padding=2,
            normalize=True,
        )

        # Convert the PyTorch tensor to a NumPy array
        grid_np = np.transpose(grid.cpu().numpy(), (1, 2, 0))

        # Display the grid using Matplotlib
        plt.figure(figsize=(20, 10))
        plt.imshow(grid_np)
        plt.axis("off")
        plt.savefig("imagenet_augmentations_paired.jpg")
        plt.close()
        sys.exit(1)
