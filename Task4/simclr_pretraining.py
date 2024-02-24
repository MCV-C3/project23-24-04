## Standard libraries
import os
from copy import deepcopy

## Imports for plotting
import matplotlib.pyplot as plt

plt.set_cmap("cividis")
from IPython.display import set_matplotlib_formats

set_matplotlib_formats("svg", "pdf")  # For export
import matplotlib

matplotlib.rcParams["lines.linewidth"] = 2.0
import seaborn as sns

sns.set()
## tqdm for loading bars
from tqdm import tqdm

## PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim

## Torchvision
import torchvision
from torchvision.datasets import STL10
from torchvision import transforms

# PyTorch Lightning
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
)

# Huggingface datasets
import datasets

# utils
from utils.transforms import preprocess, contrast_transforms
from utils.simclr_dataset import SimCLRDataset
from utils.simclr import SimCLR


NUM_WORKERS = 4
CHECKPOINT_PATH = "/ghome/group04/project23-24-04/Task4/pretrainings"
LOG_NAME = "SIMCLR_LOGS"


# Setting the seed
pl.seed_everything(42)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print("Device:", device)
print("Number of workers:", NUM_WORKERS)

dataset = datasets.load_dataset(
    "imagenet-1k", trust_remote_code=True
)  # , cache_dir='/mnt/data/datasets/imagenet-1k/data/')
train_dataset = SimCLRDataset(
    dataset["test"], transform=contrast_transforms, num_views=2
)

val_dataset = SimCLRDataset(
    dataset["validation"], transform=contrast_transforms, num_views=2
)


def train_simclr(batch_size, max_epochs=200, **kwargs):
    trainer = pl.Trainer(
        default_root_dir=os.path.join(CHECKPOINT_PATH, "SimCLR"),
        accelerator="gpu" if str(device).startswith("cuda") else "cpu",
        devices=1,
        max_epochs=max_epochs,
        callbacks=[
            ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc_top5"),
            LearningRateMonitor("epoch"),
            RichModelSummary(max_depth=20),
        ],
    )

    # Check whether pretrained model exists. If yes, load it and skip training
    pretrained_filename = os.path.join(CHECKPOINT_PATH, "SimCLR.ckpt")
    if os.path.isfile(pretrained_filename):
        print(f"Found pretrained model at {pretrained_filename}, loading...")
        model = SimCLR.load_from_checkpoint(
            pretrained_filename
        )  # Automatically loads the model with the saved hyperparameters
    else:
        train_loader = data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            pin_memory=True,
            num_workers=NUM_WORKERS,
        )
        val_loader = data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            num_workers=NUM_WORKERS,
        )
        pl.seed_everything(42)  # To be reproducable
        model = SimCLR(max_epochs=max_epochs, **kwargs)
        trainer.fit(model, train_loader, val_loader)
        model = SimCLR.load_from_checkpoint(
            trainer.checkpoint_callback.best_model_path
        )  # Load best checkpoint after training

    return model


simclr_model = train_simclr(
    batch_size=64,
    hidden_dim=32,
    lr=5e-4,
    temperature=0.07,
    weight_decay=1e-4,
    max_epochs=200,
)
