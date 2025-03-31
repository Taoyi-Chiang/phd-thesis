#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
tokenizer.py
-------------
用於載入預訓練的 tokenizer，並加入自訂 Token 詞彙。
這對於保持古文中的固定搭配或典故詞不被拆分非常有用。
"""

import os
from transformers import BertTokenizer

def load_custom_tokens(token_file="data/custom_tokens.txt"):
    """
    從檔案讀取自訂 token，每行一個 token
    """
    if not os.path.exists(token_file):
        print(f"找不到 {token_file}，將不添加自訂 token。")
        return []
    
    with open(token_file, "r", encoding="utf-8") as f:
        tokens = [line.strip() for line in f if line.strip()]
    return tokens

def update_tokenizer(model_name="bert-ancient-chinese", token_file="data/custom_tokens.txt", save_path="tokenizer_updated"):
    """
    載入預訓練 tokenizer，加入自訂 tokens，並儲存更新後的 tokenizer。
    """
    print("載入 tokenizer:", model_name)
    tokenizer = BertTokenizer.from_pretrained(model_name)
    
    custom_tokens = load_custom_tokens(token_file)
    if custom_tokens:
        print("自訂 token 數量：", len(custom_tokens))
        num_added = tokenizer.add_tokens(custom_tokens)
        print(f"成功加入 {num_added} 個 token。")
    else:
        print("沒有自訂 token 加入。")
    
    # 儲存更新後的 tokenizer
    tokenizer.save_pretrained(save_path)
    print("更新後的 tokenizer 儲存至：", save_path)
    return tokenizer

if __name__ == "__main__":
    update_tokenizer()
