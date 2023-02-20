import argparse
import glob
import os
import json
import time
import logging
import random
import re
from itertools import chain
from string import punctuation

import pandas as pd
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from sklearn.model_selection import train_test_split
from termcolor import colored
import textwrap

from dataset import KPMG_QA_DataModule

from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import torch.nn.functional as F
from torchmetrics import functional as FM

from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5TokenizerFast,
    get_cosine_schedule_with_warmup,
)

import joblib
import wandb
from pytorch_lightning.loggers import WandbLogger

wandb.login()
wandb_logger = WandbLogger()

parser = argparse.ArgumentParser(description='KPMG_Qestion_Answering')

parser.add_argument('--checkpoint_path',
                    type=str,
                    help='checkpoint path')
parser.add_argument('--ckpt', type=str)

class ArgsBase():
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        parser.add_argument('--data_file',
                            type=str,
                            default='/home/hanjuncho/kpmg_t5',
                            help='data file')

        parser.add_argument('--batch_size',
                            type=int,
                            default=2,
                            help='')

        parser.add_argument('--source_max_len',
                            type=int,
                            default=2048,
                            help='source max seq len')

        parser.add_argument('--target_max_len',
                            type=int,
                            default=32,
                            help='target max seq len')
        return parser

class Base(pl.LightningModule):
    def __init__(self, hparams, trainer, **kwargs) -> None:
        super(Base, self).__init__()
        self.save_hyperparameters(hparams)
        self.trainer = trainer

    @staticmethod
    def add_model_specific_args(parent_parser):
        # add model specific args
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)

        parser.add_argument('--batch-size',
                            type=int,
                            default=4,
                            help='batch size for training (default: 4)')

        parser.add_argument('--lr',
                            type=float,
                            default=1e-5,
                            help='The initial learning rate')

        parser.add_argument('--warmup_ratio',
                            type=float,
                            default=0.1,
                            help='warmup ratio')

        parser.add_argument('--model_path',
                            type=str,
                            default=None,
                            help='kpmg_t5 model path')
        return parser

    def setup_steps(self, stage=None):
        # NOTE There is a problem that len(train_loader) does not work.
        # After updating to 1.5.2, NotImplementedError: `train_dataloader` · Discussion #10652 · PyTorchLightning/pytorch-lightning https://github.com/PyTorchLightning/pytorch-lightning/discussions/10652
        train_loader = self.trainer._data_connector._train_dataloader_source.dataloader()
        return len(train_loader)

    def configure_optimizers(self):
        # Prepare optimizer
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.hparams.lr, correct_bias=False)

        num_workers = self.hparams.num_workers

        data_len = self.setup_steps(self)
        logging.info(f'number of workers {num_workers}, data length {data_len}')
        num_train_steps = int(data_len / (self.hparams.batch_size * num_workers) * self.hparams.max_epochs)
        logging.info(f'num_train_steps : {num_train_steps}')
        num_warmup_steps = int(num_train_steps * self.hparams.warmup_ratio)
        logging.info(f'num_warmup_steps : {num_warmup_steps}')
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)
        lr_scheduler = {'scheduler': scheduler, 
                        'monitor': 'loss', 'interval': 'step',
                        'frequency': 1}
        return [optimizer], [lr_scheduler]

class KPMG_QA_Model(Base):
    def __init__(self, hparams, trainer=None, **kwargs):
        super(KPMG_QA_Model, self).__init__(hparams, trainer, **kwargs)
        self.model = T5ForConditionalGeneration.from_pretrained('paust/pko-t5-base', return_dict=True)
        self.model.train()

    def forward(self, input_ids, attention_mask, labels=None):
        output = self.model(
            input_ids = input_ids,
            attention_mask = attention_mask,
            labels = labels
        )

        return output.loss, output.logits
    
    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        
        prob = F.softmax(outputs, dim=-1)
        preds = torch.argmax(prob, dim=-1)

        acc = FM.accuracy(task="multiclass", num_classes=outputs.shape[-1], preds=preds, target=labels)
        metrics = {"val_acc": acc, "val_loss":loss}
        self.log_dict(metrics)
        return metrics
    
    def validation_step_end(self, val_step_outputs):
        val_acc = val_step_outputs["val_acc"].cpu()
        val_loss = val_step_outputs["val_loss"].cpu()

        self.log("validation_acc", val_acc, prog_bar=True, sync_dist=True)
        self.log("validation_loss", val_loss, prog_bar=True, sync_dist=True)
        
    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        loss, outputs = self(input_ids, attention_mask, labels)
        
        prob = F.softmax(outputs, dim=-1)
        preds = torch.argmax(prob, dim=-1)
        
        acc = FM.accuracy(task="multiclass", num_classes=outputs.shape[-1], preds=preds, target=labels)
        metrics = {"test_acc": acc, "test_loss":loss}
        self.log_dict(metrics, on_epochs=True)
        return metrics
    
if __name__ == '__main__':
    pl.seed_everything(2023)
    parser = Base.add_model_specific_args(parser)
    parser = ArgsBase.add_model_specific_args(parser)
    parser = KPMG_QA_DataModule.add_model_specific_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    tokenizer = T5TokenizerFast.from_pretrained('paust/pko-t5-base')
    args = parser.parse_args()
    logging.info(args)

    original_data = joblib.load(args.data_file + "/final")
    data_df = pd.DataFrame(original_data)

    dm = KPMG_QA_DataModule(data_df,
                        tokenizer,
                        batch_size=args.batch_size,
                        source_max_len=args.source_max_len,
                        target_max_len=args.target_max_len,
                        num_workers=args.num_workers)
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val_loss',
                                                       dirpath=args.default_root_dir,
                                                       filename='model_chp/{epoch:02d}-{val_loss:.3f}',
                                                       verbose=True,
                                                       save_last=True,
                                                       mode='min',
                                                       save_top_k=3)

    lr_logger = pl.callbacks.LearningRateMonitor()
    trainer = pl.Trainer.from_argparse_args(args, logger=wandb_logger,
                                            callbacks=[checkpoint_callback, lr_logger])

    model = KPMG_QA_Model(args, trainer)

    if args.ckpt:
        trainer.fit(model, dm, ckpt_path=args.ckpt)
    else:
        trainer.fit(model, dm)

    result = trainer.test(model, dm.test_dataloader())
    print(result)