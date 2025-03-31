# Allusion-BERT Fine-Tuning 子專案

本子專案為論文中的微調模型部分，目標為訓練一個能辨識古文中典故的模型。

## 功能目標
- 使用預訓練模型 `bert-ancient-chinese` 為基礎
- 對古文語料進行 masked language model 或序列標註微調
- 最終輸出一個可辨識典故語句位置的模型

## 語料來源
- 古文語料：xxx典籍（具體來源）
- 典故對照資料：自建標註資料集，共 xx 筆，格式為 JSONL/CSV

## 訓練指令
```bash
python model/train.py --config config.yaml
