# src/model_training.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

def load_data():
    """Загрузка данных после EDA"""
    try:
        # Пробуем загрузить обработанные данные из results/
        data = pd.read_csv('results/processed_data.csv')
        print("Загружены обработанные данные из results/")
    except FileNotFoundError:
        try:
            # Пробуем загрузить из data/processed/
            data = pd.read_csv('data/processed_data.csv')
            print("Загружены обработанные данные из data/")
        except FileNotFoundError:
            # Загружаем исходные данные
            data = pd.read_csv('data/raw_data.csv')
            print("Загружены исходные данные")
            
            # Базовая предобработка
            # Если есть числовые столбцы с пропусками
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if data[col].isnull().sum() > 0:
                    data[col].fillna(data[col].median(), inplace=True)
            
            # Кодирование целевой переменной (предполагаем, что она называется 'Churn')
            if 'y' in data.columns:
                data['y'] = data['y'].map({'Yes': 1, 'No': 0})
            
            # Удаляем ID столбцы если есть
            id_columns = ['customerID', 'id', 'ID', 'customer_id']
            for col in id_columns:
                if col in data.columns:
                    data = data.drop(col, axis=1)
            
            # One-Hot Encoding для категориальных признаков
            categorical_cols = data.select_dtypes(include=['object']).columns.tolist()
            # Убираем целевую переменную из категориальных
            if 'y' in categorical_cols:
                categorical_cols.remove('y')
            
            if categorical_cols:
                data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
                print(f"Закодированы категориальные признаки: {categorical_cols}")
            
            # Сохраняем обработанные данные
            os.makedirs('results', exist_ok=True)
            data.to_csv('results/processed_data.csv', index=False)
            print("Обработанные данные сохранены в results/processed_data.csv")
    
    print(f"Данные загружены: {data.shape}")
    print(f"Колонки: {list(data.columns)}")
    return data

def train_models():
    """Обучение моделей машинного обучения"""
    print("=== ЗАГРУЗКА ДАННЫХ ===")
    data = load_data()
    
    # Проверяем, есть ли целевая переменная
    if 'y' not in data.columns:
        # Если нет 'Churn', ищем альтернативные названия
        target_candidates = ['y', 'target', 'is_y', 'Attrition']
        target_found = False
        for candidate in target_candidates:
            if candidate in data.columns:
                data = data.rename(columns={candidate: 'y'})
                target_found = True
                break
        
        if not target_found:
            raise ValueError("Не найдена целевая переменная. Ожидается столбец 'y' или аналогичный")
    
    # Разделяем на признаки и целевую переменную
    X = data.drop('y', axis=1)
    y = data['y']
    
    print(f"Признаки: {X.shape}, Целевая переменная: {y.shape}")
    print(f"Баланс классов:\n{y.value_counts(normalize=True)}")
    
    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Сохранение данных тестовых выборок 
    X_train.to_csv("data/X_train.csv", index=False)
    X_test.to_csv("data/X_test.csv", index=False)
    y_train.to_csv("data/y_train.csv", index=False)
    y_test.to_csv("data/y_test.csv", index=False)

    
    print("=== ОБУЧЕНИЕ МОДЕЛЕЙ ===")
    
    # Модель 1: Logistic Regression
    print("1. Обучение Logistic Regression...")
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train, y_train)
    
    # Модель 2: Random Forest
    print("2. Обучение Random Forest...")
    rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
    rf_model.fit(X_train, y_train)
    
    # Сохранение моделей
    os.makedirs('models', exist_ok=True)
    joblib.dump(lr_model, 'models/logistic_regression.pkl')
    joblib.dump(rf_model, 'models/random_forest.pkl')
    print("Модели сохранены в папку models/")
    
    # Оценка моделей
    print("\n=== ОЦЕНКА МОДЕЛЕЙ ===")
    
    models = {
        'Logistic Regression': lr_model,
        'Random Forest': rf_model
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n--- {name} ---")
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Метрики
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        print(f"AUC-ROC: {roc_auc_score(y_test, y_pred_proba):.4f}")
        
        # Матрица ошибок
        cm = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:")
        print(cm)
        
        results[name] = {
            'model': model,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'auc_roc': roc_auc_score(y_test, y_pred_proba)
        }
    
    # Визуализация важности признаков для Random Forest
    if len(X.columns) > 0:
        plt.figure(figsize=(12, 8))
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=True)
        
        # Берем топ-15 или все если меньше
        top_features = feature_importance.tail(min(15, len(feature_importance)))
        
        plt.barh(top_features['feature'], top_features['importance'])
        plt.title('Top Important Features - Random Forest')
        plt.xlabel('Importance')
        plt.tight_layout()
        
        # Сохраняем график
        os.makedirs('results', exist_ok=True)
        plt.savefig('results/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.savefig('results/feature_importance.pdf', bbox_inches='tight')
        print("График важности признаков сохранен в results/feature_importance.png")
        plt.show()
    
    # Сравнение моделей
    print("\n=== СРАВНЕНИЕ МОДЕЛЕЙ ===")
    comparison_df = pd.DataFrame({
        'Model': list(results.keys()),
        'AUC-ROC': [results[name]['auc_roc'] for name in results.keys()]
    }).sort_values('AUC-ROC', ascending=False)
    
    print(comparison_df)
    
    # Сохраняем результаты сравнения
    comparison_df.to_csv('results/model_comparison.csv', index=False)
    print("Сравнение моделей сохранено в results/model_comparison.csv")
    
    return results, X_test, y_test

if __name__ == "__main__":
    try:
        results, X_test, y_test = train_models()
        print("\n Обучение моделей завершено успешно!")
        print("Следующие шаги:")
        print("1. Посмотри графики в папке results/")
        print("2. Проверь метрики моделей выше")
        print("3. Модели сохранены в папке models/")
    except Exception as e:
        print(f"\n Ошибка: {e}")
        print("Проверь структуру данных и названия столбцов")