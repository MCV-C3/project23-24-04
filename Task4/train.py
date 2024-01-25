import torch
import pytorch_lightning as pl
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.lr_monitor import LearningRateMonitor
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import LearningRateFinder, BatchSizeFinder, RichModelSummary
from lightning.pytorch.tuner import Tuner


from utils.data import MITDataModule
from utils.transforms import albumentations_transform
from utils.model import MITClassifier


BATCH_SIZE = 128
DATA_ROOT = '/home/gherodes/projects/tf_test/dataset/MIT_small_train_1'
IMG_SIZE = 256
EPOCHS = 100
GPUS = 0
CLASSES = ['coast','forest','highway','inside_city','mountain','Opencountry','street','tallbuilding']
NUM_CLASSES = len(CLASSES)

LEARNING_RATE = 0.0001
MIXUP = True
MIXUP_ALPHA = 0.2

CUTMIX = True
CUTMIX_ALPHA = 0.2

dm = MITDataModule(batch_size=BATCH_SIZE,
                   root_dir=DATA_ROOT,
                   dims=(IMG_SIZE, IMG_SIZE),
                   transforms=albumentations_transform(),
                   sampler='wrs'
                   )



model = MITClassifier(learning_rate=LEARNING_RATE, 
                          num_classes=NUM_CLASSES, 
                          class_names=CLASSES,
                          mixup=MIXUP, 
                          mixup_alpha=MIXUP_ALPHA,
                          cutmix=CUTMIX,
                          cutmix_alpha=CUTMIX_ALPHA)

# checkpointer
checkpointer = ModelCheckpoint(
        monitor='val_acc',
        save_top_k=1,
        mode='max',
        save_weights_only=True
    )

# lr_monitor
learningrate_monitor = LearningRateMonitor(
        logging_interval='step'
    )

# early stopping
early_stopping = EarlyStopping(
        monitor=('val_acc'),
        min_delta=0.00,
        patience=10,
        mode='max')


# Run learning rate finder
lr_finder = LearningRateFinder(min_lr=0.00001, 
                               max_lr=0.01, 
                               num_training_steps=100, 
                               mode='exponential', 
                               early_stop_threshold=4.0, 
                               update_attr=True, 
                               attr_name='learning_rate')

# batch_size_finder = BatchSizeFinder(mode='power', 
#                                     steps_per_trial=3, 
#                                     init_val=32, 
#                                     max_trials=25, 
#                                     batch_arg_name='batch_size')

rich_summary = RichModelSummary(max_depth=20)

# training
trainer = pl.Trainer(
        max_epochs=EPOCHS,
        devices=1,
        accelerator='gpu',
        num_nodes=1, 
        precision=32, 
        log_every_n_steps=1, 
        callbacks=[ checkpointer, learningrate_monitor, early_stopping, rich_summary], # batch_size_finder, lr_finder, 
        num_sanity_val_steps=0,
    )

trainer.fit(model=model, datamodule=dm)



