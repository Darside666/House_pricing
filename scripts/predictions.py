import os
import pandas as pd
import joblib
from data_processing import preprocess_data

# Установка пути к проекту
project_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(project_dir, '../data/train.csv')  # Используем данные для теста
models_dir = os.path.join(project_dir, '../models')
model_path = os.path.join(models_dir, 'house_price_model.pkl')

# Проверка наличия данных
if not os.path.exists(data_path):
    raise FileNotFoundError(f"Файл данных не найден по пути: {data_path}")

# Проверка наличия модели
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Файл модели не найден по пути: {model_path}")

# Загрузка данных
print("Загрузка данных...")
data = pd.read_csv(data_path)

# Предобработка данных
print("Предобработка данных...")
data = preprocess_data(data)

# Удаление целевой переменной, если она есть
if 'SalePrice' in data.columns:
    data = data.drop('SalePrice', axis=1)

# Загрузка модели
print("Загрузка модели...")
model = joblib.load(model_path)

# Предсказания
print("Выполнение предсказаний...")
predictions = model.predict(data)

# Вывод результатов
print("Предсказания завершены!")
print(predictions)
