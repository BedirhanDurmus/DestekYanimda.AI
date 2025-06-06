# DestekYanımda.AI

## Proje Hakkında
DestekYanımda.AI, elektronik cihazların fiyat tahminlemesi yapan bir yapay zeka projesidir. Laptop, tablet ve mobil telefon gibi cihazların farklı ülkelerdeki fiyatlarını tahmin ederek kullanıcılara yardımcı olur.

## Özellikler
- Laptop, tablet ve mobil telefon fiyat tahminlemesi
- 5 farklı ülke için fiyat tahmini (Pakistan, Hindistan, Çin, ABD, Dubai)
- Detaylı veri analizi ve görselleştirme
- Makine öğrenmesi modelleri ile tahminleme
- PostgreSQL veritabanı entegrasyonu

## Kullanılan Teknolojiler
- Python 3.8+
- Scikit-learn
- CatBoost
- XGBoost
- Pandas
- NumPy
- PostgreSQL


## Kurulum

### Gereksinimler
- Python 3.8 veya üzeri
- PostgreSQL 12 veya üzeri
- pip (Python paket yöneticisi)

### Adımlar

1. Depoyu klonlayın:
```bash
git clone https://github.com/kullaniciadi/destekYanimda.AI.git
cd destekYanimda.AI
```

2. Sanal ortam oluşturun ve aktifleştirin:
```bash
python -m venv venv
# Windows için:
venv\Scripts\activate
# Linux/Mac için:
source venv/bin/activate
```

3. Bağımlılıkları yükleyin:
```bash
pip install -r requirements.txt
```

4. Veritabanını yapılandırın:
- Veritabanı adı: veritabani_adi
- Kullanıcı adı: kullanici_adi
- Şifre: sifre

5. Uygulamayı başlatın:
```bash
python app.py
```

## Kullanım

### Desteklenen Cihazlar
- Laptop
- Tablet
- Mobil Telefon

### Tahmin Özellikleri
Her cihaz için aşağıdaki özellikleri kullanabilirsiniz:
- RAM
- İşlemci
- Ekran Boyutu
- Depolama Kapasitesi
- Kamera Özellikleri
- Batarya Kapasitesi

## Katkıda Bulunma
1. Bu depoyu fork edin
2. Yeni bir branch oluşturun (`git checkout -b feature/yeniOzellik`)
3. Değişikliklerinizi commit edin (`git commit -am 'Yeni özellik: Açıklama'`)
4. Branch'inizi push edin (`git push origin feature/yeniOzellik`)
5. Pull Request oluşturun


# DestekYanımda.AI - Ürün Gereksinimleri Dokümanı (PRD)

## 1. Ürün Genel Bakış
DestekYanımda.AI, elektronik cihazların fiyat tahminlemesi yapan bir yapay zeka projesidir. Proje, kullanıcılara laptop, tablet ve mobil telefon gibi cihazların farklı ülkelerdeki fiyatlarını tahmin etme imkanı sunar.

## 2. Hedef Kitle
- Elektronik cihaz alıcıları
- E-ticaret platformları
- Elektronik mağaza sahipleri
- Fiyat karşılaştırma siteleri
- Elektronik cihaz distribütörleri

## 3. Temel Özellikler

### 3.1 Cihaz Kategorileri
- Laptop
- Tablet
- Mobil Telefon

### 3.2 Tahmin Özellikleri
Her cihaz kategorisi için aşağıdaki özellikler kullanılacaktır:

#### Laptop
- RAM
- İşlemci
- Ekran Boyutu
- Depolama Tipi (SSD/HDD)
- Depolama Kapasitesi
- Grafik Kartı
- İşletim Sistemi

#### Tablet
- RAM
- İşlemci
- Ekran Boyutu
- Depolama Kapasitesi
- Kamera Özellikleri
- Batarya Kapasitesi
- İşletim Sistemi

#### Mobil Telefon
- RAM
- İşlemci
- Ekran Boyutu
- Depolama Kapasitesi
- Ön Kamera
- Arka Kamera
- Batarya Kapasitesi

### 3.3 Fiyat Tahmin Ülkeleri
- Pakistan (PKR)
- Hindistan (INR)
- Çin (CNY)
- Amerika Birleşik Devletleri (USD)
- Dubai (AED)

## 4. Teknik Gereksinimler

### 4.1 Veri Toplama ve İşleme
- PostgreSQL veritabanı kullanımı
- Veri temizleme ve ön işleme
- Eksik veri yönetimi
- Veri doğrulama ve doğrulama

### 4.2 Model Gereksinimleri
- Scikit-learn tabanlı modeller
- CatBoost ve XGBoost entegrasyonu
- Model performans metrikleri (R2, MSE, MAE)
- Model versiyonlama ve kayıt

### 4.3 Sistem Gereksinimleri
- Python 3.8+
- PostgreSQL 12+
- Minimum 8GB RAM
- Minimum 100GB depolama alanı

## 5. Performans Gereksinimleri
- Tahmin süresi: < 5 saniye
- Model doğruluğu: > %85
- Sistem uptime: > %99
- Veri güncelleme sıklığı: Aylık

## 6. Güvenlik Gereksinimleri
- Veritabanı erişim güvenliği
- API endpoint güvenliği
- Veri şifreleme
- Kullanıcı kimlik doğrulama

## 7. Gelecek Özellikler
- Web arayüzü
- API entegrasyonu
- Daha fazla ülke desteği
- Gerçek zamanlı fiyat güncellemeleri
- Kullanıcı geri bildirim sistemi
- Fiyat trend analizi
- Cihaz karşılaştırma özelliği

## 8. Başarı Kriterleri
- Tahmin doğruluğu
- Kullanıcı memnuniyeti
- Sistem performansı
- Veri güncelliği
- API yanıt süreleri

## 9. Kısıtlamalar
- Veri kaynağı sınırlamaları
- Model eğitim süresi
- Sistem kaynakları
- Bütçe kısıtlamaları

## 10. Riskler ve Azaltma Stratejileri
- Veri kalitesi riskleri
- Model performans riskleri
- Sistem güvenlik riskleri
- Ölçeklenebilirlik riskleri 

## Lisans
Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakın.
