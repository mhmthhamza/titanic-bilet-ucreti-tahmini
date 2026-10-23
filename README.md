# Titanic Yolcu Verisiyle Bilet Ücreti Tahmini

Marmara Üniversitesi, İstatistik Bölümü, İstatistiksel Yazılımlar dersi bireysel proje ödevi.
Hazırlayan: Mehmet Hamza Şükür

## Amaç

Titanic yolcu verisinde bilet ücretini (`Fare`) açıklayıcı değişkenlerle tahmin eden regresyon modellerini kurmak, varsayımlarını kontrol etmek ve aynı test seti üzerinde karşılaştırmak.

## Veri

- Kaynak: Kaggle Titanic yolcu verisi (`veri/Titanic-Dataset.csv`), 891 yolcu, 12 değişken.
- Ön işleme: PassengerId, Name, Ticket, Cabin ve Embarked sütunları çıkarıldı, Sex değişkeni 0-1'e dönüştürüldü, eksik yaşı olan 177 satır silindi.
- Sonuç: 714 gözlem; bağımlı değişken `Fare`, açıklayıcılar `Survived`, `Pclass`, `Age`, `SibSp`, `Parch`, `Sex`.
- Eğitim/test: %80 / %20 (571 / 143 gözlem), `random_state=42`.

## Yöntem

1. **Doğrusal modeller:** basit regresyon (Fare ~ Pclass) ve çoklu OLS.
2. **Varsayım kontrolleri:** normallik (Q-Q grafiği, Shapiro-Wilk, Jarque-Bera), otokorelasyon (Durbin-Watson), değişen varyans (artık grafiği), çoklu bağlantı (VIF).
3. **Boyut indirgeme ve cezalı regresyon:** PCR, PLS, Ridge, Lasso.
4. **Doğrusal olmayan modeller:** KNN, SVR (doğrusal ve rbf), yapay sinir ağı (MLP).
5. **Ağaç tabanlı modeller:** CART, Bagging, Random Forest, Gradient Boosting.
6. Model parametreleri (bileşen sayısı, alpha, k, C, ağaç sayısı vb.) çapraz doğrulama ve `GridSearchCV` ile seçildi; tüm modeller aynı test setinde değerlendirildi (RMSE, MAE, R²).

## Sonuçlar (test seti)

| Model | RMSE | MAE | R² |
|---|---|---|---|
| OLS | 58,44 | 25,42 | 0,1798 |
| PCR (6 bileşen) | 58,44 | 25,42 | 0,1798 |
| PLS (2 bileşen) | 58,72 | 25,72 | 0,1717 |
| Ridge | 58,40 | 25,15 | 0,1807 |
| Lasso | 58,41 | 25,27 | 0,1804 |
| KNN | 59,35 | 19,80 | 0,1539 |
| SVR (doğrusal) | 59,67 | 17,80 | 0,1448 |
| SVR (rbf) | 66,27 | 22,92 | -0,0547 |
| Yapay sinir ağı | 56,73 | 17,19 | 0,2271 |
| CART | 58,28 | 19,07 | 0,1843 |
| Bagging | 57,53 | 18,65 | 0,2049 |
| Random Forest | 57,27 | 17,18 | 0,2122 |
| Gradient Boosting | 58,01 | 17,95 | 0,1917 |

## Bulgular ve sınırlılıklar

- Hiçbir model R² = 0,23'ü geçemedi. Bilet ücretini bu 6 değişkenle tahmin etmek zor; `Fare` aşırı sağa çarpık olduğu için OLS'nin normallik ve sabit varyans varsayımları sağlanmıyor.
- Ağaç tabanlı modeller ve yapay sinir ağı, ortalama mutlak hatada (MAE) doğrusal modellerden belirgin biçimde iyi (yaklaşık 17-20, doğrusal modellerde yaklaşık 25). Bu, tekrarlanan çalıştırmalarda da değişmeyen, projenin kararlı bulgusudur. RMSE'de fark küçük; uçtaki pahalı biletler hatayı büyütüyor.
- PCR'da çapraz doğrulama tüm bileşenleri (6/6) seçtiği için PCR sonucu düz OLS ile aynıdır.
- Lasso hiçbir değişkenin katsayısını sıfırlamadı.
- Yapay sinir ağı, Bagging, Random Forest ve Gradient Boosting'de `random_state` verilmedi; bu dört modelin sonuçları kodu her çalıştırdığımızda değişir ve aralarındaki sıralama kararlı değildir. Tablodaki değerler tek bir çalıştırmanın sonucudur; başka bir çalıştırmada örneğin YSA R² 0,189, Bagging 0,156, Random Forest 0,202, Gradient Boosting 0,208 çıktı. Bu yüzden bu dört model arasında "en iyi model" seçimi yapılmamıştır. Diğer modellerin (OLS, PCR, PLS, Ridge, Lasso, KNN, SVR, CART) sonuçları her çalıştırmada aynıdır.
- Eksik yaş değerleri silindi (doldurma yapılmadı), Cabin ve Embarked kullanılmadı.

## Çalıştırma

```bash
pip install -r requirements.txt
python titanic_analiz.py
```

Kod, `# %%` hücreleriyle yazılmıştır; Spyder'da hücre hücre (Shift+Enter) da çalıştırılabilir. Grafikler `grafikler/` klasörüne kaydedilir. Yapay sinir ağı ve Random Forest parametre aramaları birkaç dakika sürebilir.

## Klasör yapısı

```
titanic_analiz.py     ana analiz kodu
requirements.txt      gerekli paketler
veri/                 Titanic-Dataset.csv
grafikler/            üretilen grafikler
belgeler/             proje notları
```
