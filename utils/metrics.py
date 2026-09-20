import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_metrics(y_true, y_pred, average="macro"):
    """
    Tính toán các chỉ số đánh giá: Accuracy, Precision, Recall, F1-score
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=average, zero_division=0
    )

    return {
        "accuracy": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }

def plot_confusion_matrix_figure(y_true, y_pred, class_names=None):
    """
    Vẽ hình biểu đồ Confusion Matrix dưới dạng Matplotlib Figure để đẩy lên WandB.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=class_names if class_names else "auto",
                yticklabels=class_names if class_names else "auto")
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    
    return fig, cm
