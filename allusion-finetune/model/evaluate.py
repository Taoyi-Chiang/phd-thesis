#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
evaluate.py
-----------
讀取 config.yaml 中的設定，載入預訓練模型及驗證資料，
並利用 Hugging Face Trainer 評估模型表現（例如 accuracy）。
"""

import os
import yaml
import argparse
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments, DataCollatorWithPadding
from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the fine-tuned model for allusion detection")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the config file")
    return parser.parse_args()

def main():
    # 讀取 config 檔案
    args = parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # 載入 tokenizer 與模型
    print("Loading tokenizer and model from", config["model_name"])
    tokenizer = BertTokenizer.from_pretrained(config["model_name"])
    model = BertForSequenceClassification.from_pretrained(config["output_dir"])  # 假設已經訓練後儲存在 output_dir

    # 載入驗證資料集（假設 JSON 格式，包含 "text" 與 "label" 欄位）
    print("Loading evaluation dataset...")
    data_files = {"validation": config["eval_file"]}
    dataset = load_dataset("json", data_files=data_files)
    
    # 定義 tokenization 函數
    def tokenize_function(examples):
        return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=config["max_length"])
    
    print("Tokenizing evaluation dataset...")
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # 定義計算評估指標的函數 (這裡計算 accuracy)
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = (predictions == labels).astype(np.float32).mean().item()
        return {"accuracy": accuracy}
    
    # 建立 TrainingArguments (僅用於評估，不做訓練)
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        per_device_eval_batch_size=config["batch_size"],
        logging_dir=os.path.join(config["output_dir"], "logs"),
    )
    
    # 建立 Trainer 物件
    trainer = Trainer(
        model=model,
        args=training_args,
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )
    
    # 執行評估
    print("Evaluating model...")
    results = trainer.evaluate()
    print("Evaluation results:")
    for key, value in results.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main()
