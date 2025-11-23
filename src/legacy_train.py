import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pickle  # <--- Tezde bahsettiğiniz güvenlik riski (Pickle)
import os

# --- LEVEL 1 ÖZELLİĞİ: Veri Versiyonlama Yok ---
# Veri bir yerden okunmuyor, kodun içinde rastgele üretiliyor.
# Bu durum "Reproducibility" (Tekrarlanabilirlik) sorununa yol açar.
np.random.seed(42)
n_samples = 100
sleep_duration = np.random.normal(7, 1.5, n_samples)
heart_rate = np.random.normal(70, 10, n_samples)
# Stres seviyesi formülü (Gürültü eklenmiş)
stress_level = 10 - (sleep_duration * 0.8) + (heart_rate * 0.05) + np.random.normal(0, 0.5, n_samples)

df = pd.DataFrame({
    'Sleep Duration': sleep_duration,
    'Heart Rate': heart_rate,
    'Stress Level': stress_level
})

# --- LEVEL 1 ÖZELLİĞİ: Manuel Süreçler ---
X = df[['Sleep Duration', 'Heart Rate']]
y = df['Stress Level']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = LinearRegression()
model.fit(X_train, y_train)

# Tahmin
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

# --- LEVEL 1 SORUNU: İzlenebilirlik Yok ---
# Sonuçlar sadece ekrana yazılıyor. Terminal kapanınca veriler kaybolur.
print("------------------------------------------------")
print(f"Eğitim Tamamlandı (MANUEL MOD).")
print(f"MSE: {mse}") 
print(f"R2: {r2}")
print("------------------------------------------------")

# --- LEVEL 1 SORUNU: Güvenlik Riski ---
# Model, güvensiz olan 'pickle' formatında kaydediliyor.
# MLOps Level 3'te bunun yerine joblib veya ONNX kullanılır.
with open("legacy_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Model 'legacy_model.pkl' olarak güvensiz formatta kaydedildi.")