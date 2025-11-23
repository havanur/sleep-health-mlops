import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
import joblib  # <--- YENİ EKLENTİ
import os

def load_data():
    # DVC ile takip edilen veriyi oku
    csv_path = "data/sleep_health.csv"
    print(f"Veri okunuyor: {csv_path}")
    df = pd.read_csv(csv_path)
    return df

def train():
    mlflow.set_experiment("Sleep_Health_Stress_Prediction")
    
    with mlflow.start_run():
        # 1. Veri Yükleme
        df = load_data()
        X = df[['Sleep Duration', 'Heart Rate']]
        y = df['Stress Level']
        
        # 2. Bölme
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # 3. Eğitim
        params = {"fit_intercept": True}
        mlflow.log_params(params)
        
        print("Model eğitiliyor (Linear Regression)...")
        model = LinearRegression(**params)
        model.fit(X_train, y_train)
        
        # 4. Değerlendirme
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        
        print(f"Model Metrikleri -> MSE: {mse:.4f}, R2: {r2:.4f}")
        
        # 5. Kayıt (Logging)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("r2", r2)
        mlflow.sklearn.log_model(model, "model")
        print("Model ve metrikler MLflow'a kaydedildi.")

        # 6. API ve Docker için Modeli Dışarı Aktar (KRİTİK ADIM)
        joblib.dump(model, "model.pkl")
        print("Model 'model.pkl' olarak kaydedildi.")  # <--- Bu yazıyı görmeliyiz

if __name__ == "__main__":
    train()