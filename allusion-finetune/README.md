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
```

## 架構

```
/allusion_finetune/
├── data/
│   ├── raw/                # 原始語料（如古文語料、典故表）
│   ├── processed/          # 處理後語料（切詞、標註等）
│   └── stopwords.txt       # 停用詞表
│
├── model/
│   ├── config.json         # 模型與 tokenizer 設定
│   ├── train.py            # 微調主程式
│   ├── evaluate.py         # 評估腳本
│   ├── predict.py          # 預測腳本（未來 demo 可用）
│   └── tokenizer.py        # 自訂 tokenizer 加入典故詞
│
├── outputs/
│   ├── checkpoints/        # 微調後模型儲存點
│   └── logs/               # 訓練過程日誌
│
├── utils/
│   ├── preprocessing.py    # 資料清理與格式轉換
│   ├── dataset.py          # 定義 Dataset 類別
│   └── metrics.py          # 精準率/召回率等評估指標
│
├── requirements.txt        # 環境套件需求
├── README.md               # 子專案說明（簡述目標、語料來源、方法）
└── config.yaml             # 統一設定（模型路徑、訓練參數等）
```
