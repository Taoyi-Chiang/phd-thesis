#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
train.py
---------
基於 Hugging Face 的 Trainer 完成微調，
使用預訓練模型 `bert-ancient-chinese`（或你指定的模型），
並讀取 config.yaml 中的參數設定。

使用方式:
    python train.py --config config.yaml
"""

import os
import yaml
import argparse
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune BERT for allusion detection in ancient Chinese texts")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the config file")
    return parser.parse_args()

def main():
    # 解析命令列參數並讀取 config.yaml
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # 載入預訓練的 tokenizer 與模型
    print("Loading tokenizer and model from", config["model_name"])
    tokenizer = BertTokenizer.from_pretrained(config["model_name"])
    model = BertForSequenceClassification.from_pretrained(config["model_name"], num_labels=config["num_labels"])
    
    # 載入資料集（此處假設使用 JSON 格式的資料檔案，存放於 config 指定的路徑）
    print("Loading dataset...")
    data_files = {"train": config["train_file"], "validation": config["eval_file"]}
    dataset = load_dataset("json", data_files=data_files)
    
    # 定義一個 tokenization 函數，將資料集中的 "text" 欄位轉換為模型可接受的格式
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=config["max_length"])
    
    print("Tokenizing dataset...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # 建立資料 collator，用於 batch 內自動 padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # 定義計算評估指標的函數 (此例計算 accuracy)
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = (predictions == labels).astype(np.float32).mean().item()
        return {"accuracy": accuracy}
    
    # 設定訓練參數
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=config["batch_size"],
        evaluation_strategy="steps",
        eval_steps=config["eval_steps"],
        save_steps=config["save_steps"],
        logging_steps=config["logging_steps"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        logging_dir=os.path.join(config["output_dir"], "logs"),
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
    )
    
    # 建立 Trainer 物件
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # 開始訓練
    print("Start training...")
    trainer.train()
    
    # 儲存最終模型
    print("Saving model...")
    trainer.save_model(config["output_dir"])
    print("Training complete.")

if __name__ == "__main__":
    main()

