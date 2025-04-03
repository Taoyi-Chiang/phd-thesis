```
allusion_retrieval/
├── data/
│   ├── raw/
│   │   ├── ancient_texts.txt   # 若需要的話
│   │   └── allusions.csv       # 典故對照資料
│   └── processed/              # 輸出 NER/RE 預測的結果（例如標記了典故的文本）
├── ner/
│   ├── ner_train.py
│   ├── ner_evaluate.py
│   └── ner_predict.py
├── re/
│   ├── re_train.py
│   ├── re_evaluate.py
│   └── re_predict.py
├── utils/
│   └── utils.py               # 可能包括共用的預處理或後處理工具
├── config.yaml
├── requirements.txt
└── README.md
```
