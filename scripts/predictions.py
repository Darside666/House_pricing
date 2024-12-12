import pandas as pd
import joblib
from data_processing import preprocess_data

# Загрузка данных и модели
test = pd.read_csv('../data/test.csv')
model = joblib.load('../models/house_price_model.pkl')

# Предобработка данных
test_processed = preprocess_data(test)

# Предсказания
predictions = model.predict(test_processed)

# Сохранение результата
output = pd.DataFrame({'Id': test['Id'], 'SalePrice': predictions})
output.to_csv('../results/predictions.csv', index=False)
print("Предсказания сохранены в results/predictions.csv")
