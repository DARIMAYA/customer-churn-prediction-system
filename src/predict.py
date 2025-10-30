import pandas as pd
import numpy as np
import joblib
import sys

def predict_churn(new_data):
    """Предсказание оттока для новых данных"""
    
    # Загрузка модели (лучшая модель - Random Forest)
    model = joblib.load('models/random_forest.pkl')
    
    # Предобработка новых данных (аналогично training)
    # ... добавь предобработку если нужно
    
    # Предсказание
    prediction = model.predict(new_data)
    probability = model.predict_proba(new_data)[:, 1]
    
    results = pd.DataFrame({
        'prediction': prediction,
        'probability': probability,
        'churn_status': ['Yes' if p == 1 else 'No' for p in prediction]
    })
    
    return results

# Пример использования
if __name__ == "__main__":
    # Загрузи несколько примеров из тестовых данных
    X_test = pd.read_csv('data/X_test.csv').head(5)

    sys.stdout.reconfigure(encoding='utf-8')

    predictions = predict_churn(X_test)
    print("Предсказания оттока:")
    print(predictions)