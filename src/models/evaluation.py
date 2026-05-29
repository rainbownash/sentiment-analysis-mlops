import pandas as pd
from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score, f1_score

def evaluate_model(y_true, y_pred):
    return classification_report(y_true, y_pred)

def get_metrics(y_true, y_pred, representation, classifier_name):
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, zero_division=0
    )

    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')

    return pd.DataFrame([{
        'Representación': representation,
        'Clasificador': classifier_name,
        'Accuracy': round(accuracy, 2),
        'Precision 0': round(precision[0], 2),
        'Precision 1': round(precision[1], 2),
        'Recall 0': round(recall[0], 2),
        'Recall 1': round(recall[1], 2),
        'F1 0': round(f1[0], 2),
        'F1 1': round(f1[1], 2),
        'F1 macro': round(f1_macro, 2)
    }])