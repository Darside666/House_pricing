import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    # Удаление ненужных столбцов (например, идентификаторы)
    df = df.drop(['Id'], axis=1, errors='ignore')
    
    # Замена пропусков средними значениями
    for col in df.select_dtypes(include=['float64', 'int64']):
        df[col].fillna(df[col].mean(), inplace=True)
    
    # Преобразование категориальных данных
    df = pd.get_dummies(df, drop_first=True)
    
    # Нормализация числовых данных
    scaler = StandardScaler()
    df[df.select_dtypes(include=['float64', 'int64']).columns] = scaler.fit_transform(df.select_dtypes(include=['float64', 'int64']))
    
    return df
