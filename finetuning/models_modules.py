import pytorch_lightning as pl
from pytorch_lightning.callbacks.finetuning import BaseFinetuning

from transformers import ViTForImageClassification

import torch
from torch import nn, optim
import sklearn.metrics as metrics
from timm import create_model, list_models
import numpy as np

from datetime import datetime

class ImageClassifier(pl.LightningModule):
    def __init__(self, num_classes, lr=0.0001, model_name="vit_base_patch16_224", pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        self.lr = lr

        self.model = create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        #self.model = ViTForImageClassification.from_pretrained(model_name, num_labels=num_classes, ignore_mismatched_sizes=True)

        self.predict_start = None

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits, y)
        self.log('train_loss', loss)
        return {"loss": loss}
    
    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.logger.experiment.add_scalar("Loss/Train",
                                            avg_loss,
                                            self.current_epoch)
        nb_trained_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad) 
        lrs = [group["lr"] for group in self.optimizers().param_groups]
        print(f"Learning rates : {lrs}")
        print(f"# trained paremeters: {nb_trained_parameters}")

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits, y)
        self.log('val_loss', loss)
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()
        self.log('val_acc', acc)
        return {'val_loss': loss, "val_acc": acc}
    
    def validation_epoch_end(self, validation_step_outputs):
        avg_loss = torch.stack([x['val_loss'] for x in validation_step_outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in validation_step_outputs]).mean()
        self.log('val_loss', avg_loss)
        self.log("val_acc", avg_acc)
        print(f"\n\tEpoch {self.current_epoch}: val loss = {avg_loss}")
        self.logger.experiment.add_scalar("Loss/Val",
                                            avg_loss,
                                            self.current_epoch)
        self.logger.experiment.add_scalar("Acc/Val",
                                            avg_acc,
                                            self.current_epoch)

    def test_step(self, batch, batch_idx):
        if self.predict_start is None:
            self.predict_start = datetime.now()
        x, y = batch
        logits = self(x)
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()
        self.log('test_acc', acc)

        if self.num_classes == 2:
            probs = logits.sigmoid()[:, 1]
            return {"test_acc": acc, "probs": probs, "expected": y}
            
        return {"test_acc": acc}
    
    def test_epoch_end(self, test_outputs):
        end_test = datetime.now()
        test_time = (end_test-self.predict_start).total_seconds()
        self.log('test_time', test_time)
        print(f"\n\tTest time = {test_time} sec")

        avg_acc = torch.stack([x['test_acc'] for x in test_outputs]).mean()
        self.log('test_acc', avg_acc)
            
        if self.num_classes == 2:
            probs = torch.cat([x['probs'] for x in test_outputs]).detach().cpu().numpy()
            y_test = torch.cat([x['expected'] for x in test_outputs]).detach().cpu().numpy()
            
            roc_auc = metrics.roc_auc_score(y_test, probs)
            self.log('test_roc_auc', roc_auc)

    
    def configure_optimizers(self):
        # Return AdamW
        return optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.lr)
    

class FinetuningCallback(BaseFinetuning):

    def __init__(self, start_epochs = 0, new_lr = 0.00001, train_bn = False):
        self.start_epochs = start_epochs
        self.new_lr = new_lr
        self.train_bn = train_bn
        self._restarting = False
        self._internal_optimizer_metadata = {}

    def freeze_before_training(self, pl_module: pl.LightningModule):
        layers = list(pl_module.model.children())[:-1]
        for l in layers:
            self.freeze(l, train_bn=self.train_bn)

    def finetune_function(self, pl_module: pl.LightningModule, epoch: int, optimizer, opt_idx=0):
        if epoch == self.start_epochs:
            # unfreeze all layers after start_epochs
            self.unfreeze_and_add_param_group(modules=pl_module, optimizer=optimizer, train_bn=self.train_bn, lr=self.new_lr)
            for g in optimizer.param_groups:
                g['lr'] = self.new_lr

print(list_models(pretrained=True))
