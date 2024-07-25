import wandb
import torchvision
from torch import nn
import torch
import pytorch_lightning as pl
from utils import exists
import wandb
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
import io
from PIL import Image
import os


class GlobalStreetScapesClassificationTrainer(pl.LightningModule):
    
    def __init__(self, 
                 lr=0.0001,
                 pretrained=True,
                 weight=None,
                 num_classes=None, 
                 class_mapping=None,
                 model='maxvit_t',
                 **kwargs):
        
        super().__init__()
        self.lr = lr
        self.class_mapping = class_mapping
        
        if pretrained:
            self.model = torchvision.models.maxvit_t(weights=torchvision.models.MaxVit_T_Weights.DEFAULT,progress=True)
        else:
            print(f'\n \n Using untrained model: {model}')
            self.model = torchvision.models.maxvit_t(weights=None,progress=True)
        self.model.classifier[-1] = nn.Linear(in_features=512,out_features=num_classes)
        
        
        if exists(weight):
            self.loss_fn = nn.CrossEntropyLoss(weight=weight)
        else:
            self.loss_fn = nn.CrossEntropyLoss()

        if kwargs:
            self.kwargs = kwargs['kwargs']
            print(f"kwargs: {self.kwargs}")

        # Best metrics
        self.best_val_acc = 0
        self.best_test_acc = 0
        self.best_val_metrics = {'precision': 0, 'recall': 0, 'f1': 0}
        
        self.all_preds = None
        self.all_targets = None
        
        
    def setup(self, stage=None):

        # Initialize tensors here when self.device is available
        if self.all_preds is None or self.all_targets is None:
            self.all_preds = torch.tensor([], dtype=torch.long, device=self.device)
            self.all_targets = torch.tensor([], dtype=torch.long, device=self.device)

        
    def on_save_checkpoint(self, checkpoint):
        # Add your attribute to the checkpoint dictionary
        checkpoint['class_mapping'] = self.class_mapping
        return checkpoint

    def on_load_checkpoint(self, checkpoint):
        # Set the attribute from the checkpoint
        self.class_mapping = checkpoint.get('class_mapping', None)
        assert self.class_mapping is not None, "Class mapping not found in checkpoint"
    
    def forward(self, image):
        
        logits = self.model(image)

        return logits
        
    def training_step(self, batch,batch_idx):
        images,targets = batch[0],batch[1]
        logits = self.forward(images)
        loss = self.loss_fn(logits, targets)
        acc = sum((logits.softmax(dim=-1).argmax(dim=-1)==targets)/len(targets))
        
        prec,recall,f1 = self.compute_performance_measures(logits,targets)

        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_prec', prec,  on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_recall', recall,  on_step=False, on_epoch=True, prog_bar=True)
        self.log('train_f1', f1,  on_step=False, on_epoch=True, prog_bar=True)
    
        
        return {"loss": loss, "train_acc": acc}

    def validation_step(self, batch,batch_idx):
        images,targets = batch[0],batch[1]
        logits = self.forward(images)
        loss = self.loss_fn(logits, targets)
        acc = sum((logits.softmax(dim=-1).argmax(dim=-1)==targets)/len(targets))
        
        prec,recall,f1 = self.compute_performance_measures(logits,targets)

        self.log('validation_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('validation_acc', acc, on_step=False, on_epoch=True, prog_bar=True)
        self.log('validation_prec', prec,  on_step=False, on_epoch=True, prog_bar=True)
        self.log('validation_recall', recall,  on_step=False, on_epoch=True, prog_bar=True)
        self.log('validation_f1', f1,  on_step=False, on_epoch=True, prog_bar=True)
    

        return {"val_loss": loss, "val_acc": acc}

    def test_step(self, batch,batch_idx):
        images,targets = batch[0],batch[1]
        logits = self.forward(images)
        loss = self.loss_fn(logits, targets)
        acc = sum((logits.softmax(dim=-1).argmax(dim=-1)==targets)/len(targets))
        
        
        prec,recall,f1 = self.compute_performance_measures(logits,targets)
        
        self.log('test_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        self.log('test_prec', prec, on_step=True, on_epoch=True, prog_bar=True)
        self.log('test_recall', recall, on_step=True, on_epoch=True, prog_bar=True)
        self.log('test_f1', f1, on_step=True, on_epoch=True, prog_bar=True)
        
        
        # Confusion matrix metrics
        preds = logits.softmax(dim=-1).argmax(dim=-1)
        
        preds = preds.to(self.device)
        targets = targets.to(self.device)
    
        # Note: For some reason this is necessary. Potentially some DDP issue?
        # Ensure that self.all_preds and preds are on the same device
        if self.all_preds.device != preds.device:
            self.all_preds = self.all_preds.to(preds.device)

        if self.all_targets.device != targets.device:
            self.all_targets = self.all_targets.to(targets.device)
        
        self.all_preds = torch.cat((self.all_preds, preds), dim=0)
        self.all_targets = torch.cat((self.all_targets, targets), dim=0)

        

        return {"test_loss": loss, "test_acc": acc}

    def on_validation_epoch_end(self):
        avg_val_acc = self.trainer.callback_metrics.get('validation_acc')
        avg_val_prec = self.trainer.callback_metrics.get('validation_prec')
        avg_val_recall = self.trainer.callback_metrics.get('validation_recall')
        avg_val_f1 = self.trainer.callback_metrics.get('validation_f1')
        
        if avg_val_acc:    
            if avg_val_acc > self.best_val_acc:
                self.best_val_acc = avg_val_acc
                self.log('best_val_acc', self.best_val_acc, prog_bar=True)
            if avg_val_prec > self.best_val_metrics.get('precision'):
                self.best_val_metrics['precision'] = avg_val_prec
                self.log('best_val_prec', self.best_val_metrics.get('precision'), prog_bar=True)
            if avg_val_recall > self.best_val_metrics.get('recall'):
                self.best_val_metrics['recall'] = avg_val_recall
                self.log('best_val_recall', self.best_val_metrics.get('recall'), prog_bar=True)
            if avg_val_f1 > self.best_val_metrics.get('f1'):
                self.best_val_metrics['f1'] = avg_val_f1
                self.log('best_val_f1', self.best_val_metrics.get('f1'), prog_bar=True)


        out_dict = {'attribute': self.kwargs.get('attribute'),
                    'weighted': self.kwargs.get('weighted'),
                    'weighting_strategy': self.kwargs.get('weighting_strategy'),
                    'kfold_validation': self.kwargs.get('kfold'),
                    'best_validation_accuracy': self.best_val_acc,
                    'best_val_precision': self.best_val_metrics.get('precision'),
                    'best_val_recall': self.best_val_metrics.get('recall'),
                    'best_val_f1': self.best_val_metrics.get('f1'),
                }
        idx = 0 if self.kwargs.get('kfold') else self.kwargs.get('kfold')
        df = pd.DataFrame([out_dict], index=[idx]) # specify index as 0
        table = wandb.Table(data=df)

    def on_test_end(self):
        
        #Log the confusion matrix
        tgts = self.all_targets.cpu().numpy()
        preds = self.all_preds.cpu().numpy()
        cm = confusion_matrix(tgts, preds)

        fig, ax = plt.subplots(figsize=(10, 10))
        ConfusionMatrixDisplay(cm,display_labels=self.class_mapping.keys()).plot(ax=ax)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        cm_image = Image.open(buf)

        self.logger.experiment.log({"confusion_matrix": [wandb.Image(cm_image, caption="Confusion Matrix")]})
        buf.close()
        
 
        avg_test_acc = self.trainer.callback_metrics.get('test_acc')
        avg_test_prec = self.trainer.callback_metrics.get('test_prec')
        avg_test_recall = self.trainer.callback_metrics.get('test_recall')
        avg_test_f1 = self.trainer.callback_metrics.get('test_f1')
        
        assert avg_test_acc, "No test accuracy found. Did you run trainer.test()?"


        if self.kwargs:
            out_dict = {'uuid': self.kwargs.get('uuid'),
                        'attribute': self.kwargs.get('attribute'),
                        'weighted': self.kwargs.get('weighted'),
                        'weighting_strategy': self.kwargs.get('weighting_strategy'),
                        'kfold_validation': self.kwargs.get('kfold'),
                        'best_test_accuracy': avg_test_acc,
                        'best_validation_accuracy': self.best_val_acc,
                        'best_test_precision': avg_test_prec,
                        'best_test_recall': avg_test_recall,
                        'best_test_f1': avg_test_f1,
                        'sklearn_test_accuracy': accuracy_score(tgts, preds),
                        'sklearn_test_precision_macro': precision_score(tgts, preds, average='macro',zero_division=0),
                        'sklearn_test_recall_macro': recall_score(tgts, preds, average='macro',zero_division=0),
                        'sklearn_test_f1_macro': f1_score(tgts, preds, average='macro',zero_division=0),
                        'sklearn_test_precision_weighted': precision_score(tgts, preds, average='weighted',zero_division=0),
                        'sklearn_test_recall_weighted': recall_score(tgts, preds, average='weighted',zero_division=0),
                        'sklearn_test_f1_weighted': f1_score(tgts, preds, average='weighted',zero_division=0),
                       }

            idx = 0 if self.kwargs.get('kfold') else self.kwargs.get('kfold')
            df = pd.DataFrame([out_dict], index=[idx+1]) # Final test run
            
            results_dir = './results'
            if not os.path.exists(results_dir):
                os.makedirs(results_dir)
            df.to_csv(f'{results_dir}/test_results_{self.kwargs.get("uuid")}.csv', index=False)
            
            table = wandb.Table(data=df)

            # Log the table to wandb
            self.logger.experiment.log({"classification_results_test": table})
        
        
    def configure_optimizers(self):
        
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
    
    
    def compute_performance_measures(self, logits, targets):
        preds = logits.softmax(dim=-1).argmax(dim=-1)

        num_classes = logits.size(1)
        precisions = []
        recalls = []
        f1_scores = []

        for c in range(num_classes):
            tp = ((preds == c) & (targets == c)).sum().item()
            fp = ((preds == c) & (targets != c)).sum().item()
            fn = ((preds != c) & (targets == c)).sum().item()

            precision = tp / (tp + fp + 1e-10)  # Avoid division by zero
            recall = tp / (tp + fn + 1e-10)
            f1 = 2 * (precision * recall) / (precision + recall + 1e-10)  

            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)

        macro_prec = sum(precisions) / num_classes
        macro_recall = sum(recalls) / num_classes
        macro_f1 = sum(f1_scores) / num_classes

        return macro_prec, macro_recall, macro_f1


