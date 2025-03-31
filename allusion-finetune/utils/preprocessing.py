#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
preprocessing.py
-----------------
將原始古文語料轉換成 JSONL 格式，便於後續訓練使用。
每行輸入文本將轉換成一個 JSON 格式的 dict，包含 "text" 與 "label" 欄位。

用法：
    python preprocessing.py --input_file data/raw/ancient_texts.txt --output_file data/processed/processed.jsonl
"""

import os
import json
import argparse

def preprocess_text(text, stopwords=None):
    """
    對輸入的文本進行預處理，可加入停用詞過濾等處理。
    目前僅做基本 strip 處理，依需求擴充。
    
    stopwords: 可選的停用詞集合（set）
    """
    text = text.strip()
    # 如果有提供停用詞，則可進一步處理（這裡僅作示範）
    if stopwords:
        for word in stopwords:
            text = text.replace(word, "")
    return text

def load_stopwords(stopwords_file):
    """
    從檔案中讀取停用詞，每行一個詞，回傳 set 集合
    """
    if not os.path.exists(stopwords_file):
        return None
    with open(stopwords_file, "r", encoding="utf-8") as f:
        stopwords = set([line.strip() for line in f if line.strip()])
    return stopwords

def process_raw_file(input_file, output_file, stopwords_file=None, default_label=0):
    """
    讀取原始文本檔，每行視為一筆資料，進行預處理後輸出成 JSONL 檔案。
    
    default_label: 若無標註資料，預設給予的標籤（例如 0 代表「不含典故」）
    """
    stopwords = load_stopwords(stopwords_file) if stopwords_file else None
    processed_samples = []
    
    with open(input_file, "r", encoding="utf-8") as fin:
        for line in fin:
            text = preprocess_text(line, stopwords)
            if text:  # 只處理非空行
                processed_samples.append({"text": text, "label": default_label})
    
    # 輸出 JSONL 檔案
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as fout:
        for sample in processed_samples:
            fout.write(json.dumps(sample, ensure_ascii=False) + "\n")
    
    print(f"預處理完成，處理後資料已儲存至 {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess raw ancient Chinese texts into JSONL format")
    parser.add_argument("--input_file", type=str, required=True, help="Path to raw text file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to output JSONL file")
    parser.add_argument("--stopwords_file", type=str, default=None, help="Path to stopwords file (optional)")
    args = parser.parse_args()
    
    process_raw_file(args.input_file, args.output_file, args.stopwords_file)
