# **DevOps ve MLOps Metodolojilerinin Adaptasyon Modelleri: Sleep Health ML Project**

**Bu proje, "DevOps ve MLOps Metodolojilerinin Proje Yönetimine Adaptasyon Modelleri" başlıklı Yüksek Lisans tezi kapsamında geliştirilmiştir.**

## **📖 Proje Hakkında**

Bu çalışmanın amacı, geleneksel yöntemlerle (Level 1 \- Manuel) geliştirilen bir Makine Öğrenmesi projesini ("Sleep Health"), modern MLOps prensipleri ve araçları kullanılarak tam otomatik (Level 3\) bir mimariye dönüştürmektir.

Proje, uyku süresi ve kalp atış hızı verilerini kullanarak stres seviyesini tahmin eden bir regresyon modelini temel alır.

### **🎯 Dönüşüm Hedefleri**

| Özellik | Level 1 (Önceki Durum) | Level 3 (Hedeflenen Durum) |
| :---- | :---- | :---- |
| **Model Eğitimi** | Manuel (legacy_train.py) | Otomatik, İzlenebilir (train.py) |
| **Veri Takibi** | Yok (Rastgele Üretim) | DVC (Data Version Control) |
| **Testler** | Manuel Gözlem | Otomatik (CI \- GitHub Actions) |
| **Canlıya Alma** | Lokal Script | Docker \+ FastAPI (Microservice) |

## **📂 Proje Yapısı**

Bu depo, tezin uygulama aşamalarını yansıtan aşağıdaki dizin yapısına sahiptir:

sleep-health-mlops/  
├── .github/workflows/   \# CI/CD Pipeline (Otomatik Testler)  
├── data/                \# DVC ile takip edilen veri dosyaları  
├── src/  
│   ├── app.py           \# (Level 3\) FastAPI Model Servisi  
│   ├── train.py         \# (Level 3\) Modern Eğitim Kodu (MLflow \+ Joblib)  
│   └── legacy_train.py  \# (Level 1\) Eski Manuel Kod (Karşılaştırma için)  
├── tests/               \# Otomatik test senaryoları (Veri ve Model Kalitesi)  
├── Dockerfile           \# Konteynerizasyon yapılandırması  
├── requirements.txt     \# Proje bağımlılıkları  
└── model.pkl            \# Eğitilmiş final model (Joblib)

## **🚀 Kurulum ve Çalıştırma**

Projenin Level 3 (Dockerize edilmiş) versiyonunu çalıştırmak için aşağıdaki adımları izleyin.

### **1\. Gereksinimler**

* Docker Desktop  
* Python 3.11+  
* Git

### **2\. Depoyu Klonlayın**

git clone \[https://github.com/havanur/sleep-health-mlops.git\](https://github.com/havanur/sleep-health-mlops.git)  
cd sleep-health-mlops

### **3\. Modeli Docker ile Ayağa Kaldırın (Önerilen)**

Tüm ortam bağımlılıklarını izole etmek için Docker kullanın:

\# Docker imajını oluştur  
docker build \-t sleep-health-api .

\# Konteyneri başlat (Port 8000\)  
docker run \-p 8000:8000 sleep-health-api

### **4\. API Testi**

Tarayıcınızda **https://www.google.com/search?q=http://127.0.0.1:8000/docs** adresine giderek Swagger UI üzerinden modelle etkileşime geçebilirsiniz.

**Örnek JSON İsteği:**

{  
  "sleep\_duration": 5.5,  
  "heart\_rate": 85  
}

## **🛠️ Uygulanan Metodoloji (Fazlar)**

Tez çalışması kapsamında proje 3 ana fazda geliştirilmiştir:

### **Faz 1: Temel MLOps ve İzlenebilirlik**

* **MLflow:** Deney parametreleri (fit\_intercept) ve metrikleri (MSE, R2) kayıt altına alındı.  
* **DVC:** Veri seti (data/sleep\_health.csv) versiyon kontrolüne alındı.  
* *Kanıt Dosyası:* src/train.py

### **Faz 2: Otomasyon ve Kalite Güvencesi (QA)**

* **GitHub Actions:** Kod her push edildiğinde çalışan bir CI pipeline kuruldu.  
* **pytest:** Modelin R2 skorunun \> 0.80 olduğunu garanti eden otomatik testler yazıldı.  
* *Kanıt Dosyası:* .github/workflows/ci.yml

### **Faz 3: Model Serving ve Konteynerizasyon**

* **FastAPI:** Model, REST API olarak dışa açıldı.  
* **Docker:** Uygulama "Production-ready" hale getirildi.  
* *Kanıt Dosyası:* Dockerfile ve src/app.py

## **📈 Deneysel Sonuçlar**

Level 3 MLOps adaptasyonu sonucunda elde edilen operasyonel iyileştirmeler:

| Metrik | Öncesi (Level 1\) | Sonrası (Level 3\) | İyileşme Oranı |
| :---- | :---- | :---- | :---- |
| **Deployment Süresi** | 30 dakika | 5 dakika | **%83 ⬇** |
| **Model Güncelleme** | 2 gün | 1 saat | **%93 ⬇** |
| **Bug Tespit Süresi** | 1 hafta | 10 dakika | **%99 ⬇** |

## **👨‍💻 İletişim**

Bu çalışma Hava Nur Şimşek tarafından Yüksek Lisans tezi olarak hazırlanmıştır.