import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
import os

# 1. Veri Hazırlığı (Tezdeki Sleep Health senaryosu simülasyonu)
def load_data():
    # Gerçek hayatta burası: data = pd.read_csv("data/sleep_health.csv")
    np.random.seed(42)
    n_samples = 100
    
    # Feature'lar: Uyku Süresi, Kalp Atış Hızı, Sistolik Tansiyon
    sleep_duration = np.random.normal(7, 1.5, n_samples)
    heart_rate = np.random.normal(70, 10, n_samples)
    
    # Hedef: Stres Seviyesi (1-10 arası) - Basit bir ilişki kuralım
    # Daha az uyku + yüksek nabız = yüksek stres
    stress_level = 10 - (sleep_duration * 0.8) + (heart_rate * 0.05) + np.random.normal(0, 0.5, n_samples)
    
    # Veriyi DataFrame'e çevir
    df = pd.DataFrame({
        'Sleep Duration': sleep_duration,
        'Heart Rate': heart_rate,
        'Stress Level': stress_level
    })
    return df

def train():
    # MLflow İzlemeyi Başlat (Tezde belirtilen Artifact Yönetimi)
    mlflow.set_experiment("Sleep_Health_Stress_Prediction")
    
    with mlflow.start_run():
        print("Veri yükleniyor...")
        df = load_data()
        
        X = df[['Sleep Duration', 'Heart Rate']]
        y = df['Stress Level']
        
        # Train/Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Parametreler
        params = {"fit_intercept": True}
        mlflow.log_params(params)
        
        print("Model eğitiliyor (Linear Regression)...")
        model = LinearRegression(**params)
        model.fit(X_train, y_train)
        
        # Tahmin ve Değerlendirme
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        print(f"Model Metrikleri -> MSE: {mse:.4f}, R2: {r2:.4f}")
        
        # Metrikleri MLflow'a kaydet
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)
        
        # Modeli kaydet (Artifact)
        mlflow.sklearn.log_model(model, "model")
        print("Model ve metrikler MLflow'a kaydedildi.")

if __name__ == "__main__":
    train()