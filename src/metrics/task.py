# src/metrics/task.py
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, precision_score, confusion_matrix
from typing import Dict, Union
import numpy as np
import json


def compute_classification_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    average: str = "macro",
    save_path: Union[str, None] = None
) -> Dict[str, float]:
    """
    Вычисляет базовые метрики качества классификации:
    Accuracy, AUC, F1, Sensitivity (Recall), Specificity.

    :param logits: сырой выход модели (до softmax), shape [B, C]
    :param targets: ground truth метки, shape [B]
    :param average: тип агрегации для multiclass ("macro" | "weighted")
    :param save_path: путь для сохранения результатов в json (опционально)
    :return: словарь метрик
    """
    y_true = targets.cpu().numpy()
    y_prob = F.softmax(logits, dim=1).detach().cpu().numpy()
    y_pred = np.argmax(y_prob, axis=1)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average=average)
    recall = recall_score(y_true, y_pred, average=average)
    precision = precision_score(y_true, y_pred, average=average)

    # Sensitivity == recall, Specificity — через confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn = np.diag(cm).sum() - np.trace(cm)  # сумма диагонали минус TP всех классов
    fp_fn = cm.sum(axis=1) + cm.sum(axis=0) - 2 * np.diag(cm)
    specificity = tn / (tn + fp_fn.sum() - tn + 1e-8)

    try:
        auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average=average)
    except ValueError:
        auc = float("nan")  # может быть недоступен, если в y_true 1 класс

    results = {
        "accuracy": acc,
        "auc": auc,
        "f1": f1,
        "sensitivity": recall,
        "specificity": specificity,
        "precision": precision
    }

    if save_path is not None:
        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)

    return results
