import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

class VisualizationError(Exception):
    pass

def plot_accuracy_loss(history, output_dir=\"outputs\"):
    \"\"\"Plot training and validation accuracy and loss.\"\"\"
    try:
        os.makedirs(output_dir, exist_ok=True)
        acc = history.history.get('accuracy', [])
        val_acc = history.history.get('val_accuracy', [])
        loss = history.history.get('loss', [])
        val_loss = history.history.get('val_loss', [])
        epochs = range(1, len(acc) + 1)

        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, acc, 'b-', label='Training Accuracy')
        plt.plot(epochs, val_acc, 'r-', label='Validation Accuracy')
        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, loss, 'b-', label='Training Loss')
        plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'accuracy_loss.png'))
        plt.close()
    except Exception as e:
        raise VisualizationError(f\"Failed to plot accuracy/loss: {e}\")

def plot_confusion_matrix(y_true, y_pred, class_names, output_dir=\"outputs\"):
    \"\"\"Plot and save confusion matrix.\"\"\"
    try:
        os.makedirs(output_dir, exist_ok=True)
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(12, 10))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'))
        plt.close()
        return cm
    except Exception as e:
        raise VisualizationError(f\"Failed to plot confusion matrix: {e}\")

def save_classification_report(y_true, y_pred, class_names, output_dir=\"outputs\"):
    \"\"\"Save classification report to text file.\"\"\"
    try:
        os.makedirs(output_dir, exist_ok=True)
        report = classification_report(y_true, y_pred, target_names=class_names)
        report_path = os.path.join(output_dir, 'classification_report.txt')
        with open(report_path, 'w') as f:
            f.write(report)
        return report
    except Exception as e:
        raise VisualizationError(f\"Failed to save classification report: {e}\")
