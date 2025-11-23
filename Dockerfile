# 1. Temel İmaj: Sürümü 3.11'e yükselttik (Local ortamla eşitledik)
FROM python:3.11-slim

# 2. Çalışma Dizini: Konteyner içinde '/app' klasöründe çalışacağız
WORKDIR /app

# 3. Bağımlılıkları Kopyala ve Yükle
# (Önce sadece requirements.txt'yi kopyalıyoruz ki Docker önbelleği (cache) kullansın)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 4. Kaynak Kodları ve Modeli Kopyala
COPY src/ src/
COPY model.pkl .

# 5. Dışarıya Açılacak Port (Bilgilendirme amaçlı)
EXPOSE 8000

# 6. Başlatma Komutu
# Uygulama ayağa kalkarken bu komutu çalıştıracak
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]