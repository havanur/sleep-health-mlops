import pandas as pd
import numpy as np
from src.train import load_data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def test_data_quality():
    """
    Tezdeki Veri Kalitesi Kontrolü :
    Veri setinin boş olmadığını ve beklenen sütunları içerdiğini test eder.
    """
    df = load_data()
    
    # 1. Veri boş olmamalı
    assert not df.empty, "Veri seti boş geldi!"
    
    # 2. Kritik sütunlar var mı?
    expected_columns = ['Sleep Duration', 'Heart Rate', 'Stress Level']
    for col in expected_columns:
        assert col in df.columns, f"Eksik sütun: {col}"

def test_model_performance():
    """
    Tezdeki Model Kalite Kontrolü :
    Modelin R2 skorunun 0.80'in üzerinde olup olmadığını test eder.
    """
    df = load_data()
    X = df[['Sleep Duration', 'Heart Rate']]
    y = df['Stress Level']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    r2 = r2_score(y_test, predictions)
    
    # Başarı Kriteri: R2 > 0.80 olmalı
    assert r2 > 0.80, f"Model performansı yetersiz! R2: {r2}"