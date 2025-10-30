# customer-churn-prediction-system
Цель: Разработать и развернуть автоматизированную систему, которая идентифицирует клиентов банка/телеком-компании с высоким риском ухода, чтобы отдел маркетинга мог предложить им персональные промо-акции (End-to-End система прогнозирования оттока клиентов с ML пайплайном)

# Customer Churn Prediction System

[![Python](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/)
[![ML](https://img.shields.io/badge/machine-learning-orange.svg)](https://scikit-learn.org/)
[![API](https://img.shields.io/badge/API-FastAPI-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/container-docker-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)

End-to-End система прогнозирования оттока клиентов с полным ML пайплайном. От данных до production-ready API.

## О проекте

Комплексное решение для предсказания оттока клиентов телеком-компании, включающее:
- **ETL-пайплайн** для обработки и обогащения данных
- **ML модель** с ROC-AUC 0.89 для прогнозирования оттока  
- **REST API** для интеграции с бизнес-системами
- **Docker контейнеризацию** для развертывания
- **Интерактивный дашборд** для визуализации результатов

## Быстрый старт

### Предварительные требования
- Python 3.9+
- Docker (опционально)

### Установка и запуск

1. **Клонируйте репозиторий**
```bash
git clone https://github.com/DARIMAYA/customer-churn-prediction-system.git
cd customer-churn-prediction-system
