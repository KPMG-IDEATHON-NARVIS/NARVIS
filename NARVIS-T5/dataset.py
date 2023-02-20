import argparse
import os
import glob
import torch
import ast
import numpy as np
import pandas as pd
from tqdm import tqdm, trange
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from functools import partial
from transformers import T5TokenizerFast
from sklearn.model_selection import train_test_split

class KPMG_QA_Dataset(Dataset):
  def __init__(
      self,
      data: pd.DataFrame,
      tokenizer: T5TokenizerFast,
      source_max_token_len: int = 2048,
      target_max_token_len: int = 8
  ):
  
    self.tokenizer = tokenizer
    self.data = data
    self.source_max_token_len = source_max_token_len
    self.target_max_token_len = target_max_token_len

  def __len__(self):
    return len(self.data)

  def __getitem__(self, index: int):
    data_row = self.data.iloc[index]

    process_question = "question: " + data_row["question"]
    process_context = "context: " + " ".join(data_row["context"]).replace("[CLS]","")

    source_encoding = self.tokenizer(
      process_question,
      process_context,  
      max_length = self.source_max_token_len,
      padding = "max_length",
      truncation = "only_second",
      return_attention_mask = True,
      add_special_tokens = True,
      return_tensors="pt"
    )

    target_encoding = self.tokenizer(
      data_row["answer"],
      max_length = self.target_max_token_len,
      padding = "max_length",
      truncation = True,
      return_attention_mask = True,
      add_special_tokens = True,
      return_tensors = "pt"
    )

    labels = target_encoding["input_ids"]
    labels[labels == 0] = -100

    return dict(
        question=data_row["question"],
        context=data_row["context"],
        answer_text = data_row["answer"],
        input_ids = source_encoding["input_ids"].flatten(),
        attention_mask = source_encoding["attention_mask"].flatten(),
        labels=labels.flatten()
    )

class KPMG_QA_DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_df: pd.DataFrame,
        tokenizer: T5TokenizerFast,
        batch_size: int = 2,
        source_max_len: int = 2048,
        target_max_len: int = 8,
        num_workers: int = 4
        ):
        super().__init__()
        self.batch_size = batch_size
        self.train_df, self.test_df = train_test_split(data_df, test_size = 0.15, random_state=2023) 
        self.tokenizer = tokenizer
        self.source_max_token_len = source_max_len
        self.target_max_token_len = target_max_len
        self.num_workers = num_workers

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = argparse.ArgumentParser(
            parents=[parent_parser], add_help=False)
        parser.add_argument('--num_workers',
                            type=int,
                            default=4,
                            help='num of worker for dataloader')
        return parser

    def setup(self, stage=None):
        self.train_dataset = KPMG_QA_Dataset(
            self.train_df,
            self.tokenizer,
            self.source_max_token_len,
            self.target_max_token_len
        )

        self.test_dataset = KPMG_QA_Dataset(
            self.test_df,
            self.tokenizer,
            self.source_max_token_len,
            self.target_max_token_len
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size = self.batch_size,
            shuffle = True,
            num_workers = self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size = self.batch_size,
            num_workers = self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size = self.batch_size,
            num_workers = self.num_workers
        )