#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
dataset.py
----------
定義自訂的 AllusionDataset 用於讀取預處理後的 JSONL 格式資料。
每一行資料應包含 "text" 與 "label" 欄位。
"""

import json
import torch
from torch.utils.data import Dataset

class AllusionDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=128):
        """
        file_path: JSONL 檔案路徑，每行是一個 JSON 格式的 dict (包含 text 與 label)
        tokenizer: Hugging Face 的 tokenizer，用來對 text 進行編碼
        max_length: 最大序列長度
        """
        self.samples = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                # 每行一筆 JSON 資料
                self.samples.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample.get("text", "")
        label = sample.get("label", 0)
        
        # 使用 tokenizer 對文本進行編碼
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors="pt"
        )
        # 將 batch 維度 squeeze 掉
        encoding = {key: val.squeeze(0) for key, val in encoding.items()}
        encoding["labels"] = torch.tensor(label, dtype=torch.long)
        return encoding

# 測試用法
if __name__ == "__main__":
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-ancient-chinese")
    dataset = AllusionDataset("data/processed/sample.jsonl", tokenizer, max_length=128)
    print("資料筆數：", len(dataset))
    print("第一筆資料：", dataset[0])
