import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

from data_processing import preprocess_data

# Загрузка данных
train = pd.read_csv('../data/train.csv')
test = pd.read_csv('../data/test.csv')

# Предобработка
train = preprocess_data(train)

# Разделение данных
X = train.drop('SalePrice', axis=1)
y = train['SalePrice']
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Обучение модели
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Оценка модели
y_pred = model.predict(X_val)
print("MAE:", mean_absolute_error(y_val, y_pred))
print("R²:", r2_score(y_val, y_pred))

# Сохранение модели
joblib.dump(model, '../models/house_price_model.pkl')
