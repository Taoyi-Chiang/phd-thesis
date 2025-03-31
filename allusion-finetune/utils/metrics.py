#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
metrics.py
----------
定義計算評估指標的函數，回傳 accuracy、precision、recall 與 F1-score。
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def compute_metrics(eval_pred):
    """
    eval_pred: tuple (logits, labels)
    回傳一個 dict，包含各項評估指標。
    """
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average="weighted", zero_division=0)
    recall = recall_score(labels, predictions, average="weighted", zero_division=0)
    f1 = f1_score(labels, predictions, average="weighted", zero_division=0)
    
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

# 測試用法
if __name__ == "__main__":
    # 模擬 logits 與 labels
    import numpy as np
    logits = np.array([[2.0, 1.0], [0.5, 1.5], [1.2, 0.8]])
    labels = np.array([0, 1, 0])
    metrics = compute_metrics((logits, labels))
    print("測試評估指標：", metrics)
