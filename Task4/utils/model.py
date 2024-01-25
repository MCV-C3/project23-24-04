import torch
import torchvision.models as models
import timm
import pytorch_lightning as pl
from torch import Tensor
import numpy as np

from torch import nn
from torch.nn.functional import cross_entropy
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, MultiStepLR
from torchmetrics.classification import MulticlassAccuracy
from sklearn.metrics import f1_score

from sklearn.metrics import confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

#from utils.mixup import mixup_criterion, mixup_data
from torchvision.transforms import v2


from torchsummary import summary



class MITClassifier(pl.LightningModule):
    def __init__(self, num_classes, class_names, learning_rate=None, mixup=None, mixup_alpha=None, cutmix=None, cutmix_alpha=None):
        super().__init__()
        backbone = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)

        # define properties
        self.classifier = nn.Linear(num_filters, num_classes)
        self.class_names = class_names
        self.softmax = nn.Softmax(dim=1)
        self.num_classes = num_classes

        self.learning_rate = learning_rate
        self.criterion = cross_entropy
        self.metric_train = MulticlassAccuracy(num_classes=num_classes, average=None)
        self.metric_val = MulticlassAccuracy(num_classes=num_classes, average=None)

        self.mixup = mixup
        self.mixup_alpha = mixup_alpha
        self.cutmix = cutmix
        self.cutmix_alpha = cutmix_alpha

        self.training_step_outputs = []
        self.validation_step_outputs = []
        
    def forward(self, x):

        # with torch.no_grad():
        #     representations = self.feature_extractor(x).flatten(1)   
        representations = self.feature_extractor(x).flatten(1)
        logits = self.classifier(representations)
        x = self.softmax(logits)

        return x
    
    def forward_intermediate(self, x, intermediate_layer=None):
        # Extract features up to the specified layer
        features = self.feature_extractor[:intermediate_layer](x)
        
        # Flatten the features for further processing
        representations = torch.flatten(features, 1)
        
        return representations


    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        #optimizer = torch.optim.NAdam(self.parameters(), lr=self.learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, momentum_decay=0.004, foreach=None)
        lr_scheduler = {
             'scheduler': CosineAnnealingLR(optimizer=optimizer, T_max=5, eta_min=1e-6),
            # 'scheduler': CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=2, T_mult=2, eta_min=1e-7),
            # 'scheduler': MultiStepLR(optimizer=optimizer, milestones=[60, 180], gamma=0.1),
            'name': 'lr_scheduler'
        }

        return [optimizer], [lr_scheduler]


    def training_step(self, batch, batch_idx):
        x, y = batch
        if self.cutmix and not self.mixup:
            operation = v2.CutMix(num_classes=self.num_classes, alpha=self.cutmix_alpha)
        if not self.cutmix and self.mixup:
            operation = v2.MixUp(num_classes=self.num_classes, alpha=self.mixup_alpha)
        if self.cutmix and self.mixup:
            cutmix = v2.CutMix(num_classes=self.num_classes, alpha=self.cutmix_alpha)
            mixup = v2.MixUp(num_classes=self.num_classes, alpha=self.mixup_alpha)
            operation = v2.RandomChoice([cutmix, mixup])
        x, y = operation(x, y)

        # # Create a 4x4 grid    
        # fig, axs = plt.subplots(4, 4, figsize=(20, 20))

        # for i, (image, label) in enumerate(zip(x,y)):
            
        #     # Calculate the row and column for the subplot
        #     row, col = divmod(i, 4)

        #     axs[row, col].imshow(image.permute(1, 2, 0).cpu().numpy())
        #     indices = torch.nonzero(label)
        #     try:
        #         axs[row, col].set_title(f'Label 1: {self.class_names[indices[0]]}\nLabel 2: {self.class_names[indices[1]]}')
        #     except:
        #         axs[row, col].set_title(f'Label 1: {self.class_names[indices[0]]}\nLabel 2: {self.class_names[indices[0]]}')

        #     axs[row, col].axis('off')
        # plt.savefig(f'cutmixes/cutmix_batch_{batch_idx}.png')
        # plt.close()

        logits = self(x)

        loss = self.criterion(logits, y)
       
        logs = {'train_loss': loss}
        self.log_dict(logs, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.training_step_outputs.append({'loss': loss, 'logits': logits, "targets": y})
        return {'loss': loss, 'logits': logits, "targets": y}
    
    def on_train_epoch_end(self): 
        #try: 
            all_preds = torch.cat([out['logits'] for out in self.training_step_outputs])
            all_targets = torch.cat([out['targets'] for out in self.training_step_outputs])
            self.metric_train(all_preds.argmax(1), all_targets.argmax(1))
            scores = self.metric_train.compute()
            score_average = torch.nanmean(scores)
        
            logs = {'train_acc': score_average}
            for ind, score in enumerate(scores):
                logs.update({f'train_acc_class_{ind}': score})
            self.log_dict(logs, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.metric_train.reset()
            self.training_step_outputs.clear()
        #except:
            #print('LR Finder Error?')


    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
       
        logs = {'val_loss': loss}
        self.log_dict(logs, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.validation_step_outputs.append({'loss': loss, 'logits': logits, "targets": y})
        return {'loss': loss, 'logits': logits, "targets": y}

    def on_validation_epoch_end(self): 

        all_preds = torch.cat([out['logits'] for out in self.validation_step_outputs])
        all_targets = torch.cat([out['targets'] for out in self.validation_step_outputs])

        cf_matrix = confusion_matrix(Tensor.cpu(all_targets), Tensor.cpu(all_preds.argmax(1)), normalize='true')
        df_cm = pd.DataFrame(cf_matrix, index = [i for i in self.class_names],
                            columns = [i for i in self.class_names])
        
        plt.figure(figsize = (12,7))       
        sn.heatmap(df_cm, annot=True)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig('cf_matrix_val.png')
        plt.close()         

        self.metric_val(all_preds.argmax(1), all_targets)
        scores = self.metric_val.compute()
        score_average = torch.nanmean(scores)
       
        logs = {'val_acc': score_average}
        for ind, score in enumerate(scores):
            logs.update({f'val_acc_class_{ind}': score})
        self.log_dict(logs, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.metric_val.reset()
        self.validation_step_outputs.clear() 

