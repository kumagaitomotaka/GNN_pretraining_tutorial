import pytorch_lightning as pl
import torch
import os
import csv


class CSVLogger(pl.Callback):
    def __init__(self, file_path, fieldnames):
        self.file_path = file_path
        self.fieldnames = fieldnames
        self._initialize_csv()

    def _initialize_csv(self):
        if not os.path.exists(self.file_path):
            with open(self.file_path, mode='w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=self.fieldnames)
                writer.writeheader()

    def on_validation_epoch_end(self, trainer, pl_module):
        # Get train and validation loss from the trainer
        train_loss = trainer.callback_metrics.get("train_loss", "N/A")
        val_loss = trainer.callback_metrics.get("val_loss", "N/A")
        if train_loss != "N/A":
            train_loss = train_loss.cpu().detach().numpy()
        if val_loss != "N/A":
            val_loss = val_loss.cpu().detach().numpy()

        # Append the current epoch, train loss, and validation loss to the CSV file
        with open(self.file_path, mode='a', newline='') as file:
            #print('File opened successfully!')
            writer = csv.DictWriter(file, fieldnames=self.fieldnames)
            writer.writerow({'epoch': trainer.current_epoch, 'train_loss': train_loss, 'val_loss': val_loss})
            
        
            
