#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
predict.py
-----------
讀取 config.yaml 中的設定，載入已訓練模型與 tokenizer，
對輸入的文本進行預測，並輸出預測結果。
"""

import argparse
import yaml
import torch
from transformers import BertTokenizer, BertForSequenceClassification

def parse_args():
    parser = argparse.ArgumentParser(description="Predict allusion in ancient Chinese text")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--text", type=str, required=True, help="Input text for prediction")
    return parser.parse_args()

def main():
    args = parse_args()
    # 讀取 config
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    # 載入 tokenizer 與模型
    tokenizer = BertTokenizer.from_pretrained(config["model_name"])
    model = BertForSequenceClassification.from_pretrained(config["output_dir"])
    model.eval()
    
    # 對輸入文本進行 tokenization
    inputs = tokenizer(args.text, return_tensors="pt", truncation=True, padding="max_length", max_length=config["max_length"])
    
    # 模型預測
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    prediction = torch.argmax(logits, dim=-1).item()
    
    # 輸出結果（假設 0 為不含典故，1 為含典故，可根據實際標籤修改）
    label_map = {0: "不含典故", 1: "含典故"}
    print("輸入文本：", args.text)
    print("預測結果：", label_map.get(prediction, f"未知標籤 {prediction}"))

if __name__ == "__main__":
    main()
