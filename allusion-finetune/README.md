# Allusion-BERT Fine-Tuning

本子專案旨在基於預訓練模型 `bert-ancient-chinese`，微調出能夠辨識古文中典故的模型。  
我們將利用古文語料與典故對照資料，進行續訓練及下游任務微調，最終達成自動檢測文本中隱含典故的目標。

## 主要內容
- **資料處理**：整理古文語料、停用詞表、自訂 Token 詞庫，以及典故對照資料的預處理與格式轉換（例如 JSONL）。
- **模型微調**：基於`Hugging Face`的 `transformers` 套件，對 `bert-ancient-chinese` 進行續訓練與任務微調。
- **評估與輸出**：提供訓練、評估腳本與預測 demo，方便產出論文所需圖表與數據。

## 目錄結構

```
/allusion_finetune/
├── data/
│   ├── raw/                # 原始語料（如古文語料、典故表）
│   │   ├── ancient_texts.txt    # 古文原始語料（示例檔案）
│   │   └── allusions.csv        # 典故對照資料（示例檔案）
│   ├── processed/          # 處理後語料（切詞、標註等）
│   └── stopwords.txt       # 停用詞表
│
├── model/
│   ├── train.py            # 微調主程式
│   ├── evaluate.py         # 評估腳本
│   ├── predict.py          # 預測腳本（未來 demo 可用）
│   └── tokenizer.py        # 自訂 tokenizer 加入典故詞
│
├── notes/
│   ├── checkpoint/        # 微調後模型儲存點
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


## 使用說明

1. **安裝依賴套件**  
   在專案根目錄下執行：
   ```bash
   pip install -r requirements.txt
   ```
2. **準備資料**
   - 將原始古文語料與典故對照資料存放於`data/raw/`。
   - 若有自訂停用詞表，請存放在`data/stopwords.txt`。
   
3. **開始訓練**
   配置`config.yaml`之後，執行：
   ```
   python model/train.py --config config.yaml
   ```

### 更多細節請參考後續文件與註解。
