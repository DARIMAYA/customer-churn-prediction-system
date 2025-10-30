import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, classification_report
import joblib

def detailed_evaluation():
    """Детальная оценка моделей"""
    
    # Загрузка моделей и тестовых данных
    lr_model = joblib.load('models/logistic_regression.pkl')
    rf_model = joblib.load('models/random_forest.pkl')
    
    X_test = pd.read_csv('data/X_test.csv')
    y_test = pd.read_csv('data/y_test.csv')
    
    models = {
        'Logistic Regression': lr_model,
        'Random Forest': rf_model
    }
    
    plt.figure(figsize=(15, 5))
    
    # ROC Curve
    plt.subplot(1, 3, 1)
    for name, model in models.items():
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc_score(y_test, y_pred_proba):.3f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves')
    plt.legend()
    
    # Precision-Recall Curve
    plt.subplot(1, 3, 2)
    for name, model in models.items():
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        plt.plot(recall, precision, label=name)
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves')
    plt.legend()
    
    # Feature Importance (только для Random Forest)
    plt.subplot(1, 3, 3)
    feature_importance = pd.DataFrame({
        'feature': X_test.columns,
        'importance': rf_model.feature_importances_
    }).sort_values('importance', ascending=True).tail(10)
    
    plt.barh(feature_importance['feature'], feature_importance['importance'])
    plt.title('Top 10 Features - Random Forest')
    plt.tight_layout()
    
    plt.savefig('results/detailed_evaluation.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    detailed_evaluation()