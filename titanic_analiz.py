# -*- coding: utf-8 -*-
"""
İSTATİSTİKSEL YAZILIMLAR PROJE ÖDEVİ (temiz sürüm)
Mehmet Hamza Şükür

Titanic yolcu verisinde bilet ücretini (Fare) tahmin eden regresyon modelleri:
basit/çoklu OLS, varsayım kontrolleri, VIF ve eğitim-test ayrımı.

Kullanım: Spyder'da hücre hücre (Shift+Enter) ya da `python titanic_analiz.py`
Girdi   : veri/Titanic-Dataset.csv
Çıktı   : grafikler/*.png
"""

# %% 0. Kütüphaneler ve ayarlar
from pathlib import Path

import matplotlib.pyplot as mt
import numpy as np
import pandas as pd
import pylab  # noqa: F401  (Q-Q grafiği için scipy'nin çizim arayüzü)
import scipy.stats as stats
import seaborn as sb
import statsmodels.api as sm
from scipy.stats import shapiro
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

SEED_BOLME = 42    # eğitim-test ayrımı
SEED_CV = 145      # çapraz doğrulama

KLASOR = Path(__file__).resolve().parent
GRAFIK = KLASOR / "grafikler"
GRAFIK.mkdir(exist_ok=True)


def grafik_kaydet(ad):
    """Aktif grafiği grafikler/ klasörüne kaydeder, sonra gösterir."""
    mt.savefig(GRAFIK / f"{ad}.png", dpi=150, bbox_inches="tight")
    mt.show()


# %% 1. VERİ ÖN İŞLEME
# Şimdi Veri Setini Yükleyelim
data_giris = pd.read_csv(KLASOR / "veri" / "Titanic-Dataset.csv")

# Verinin kısaca hikayesi şöyle:
# 15 Nisan 1912'de, RMS Titanic ilk yolculuğu sırasında, bir buzdağına çarptıktan sonra battı.
# Ne yazık ki gemideki herkese yetecek kadar cankurtaran filikası yoktu ve bu da 2224 yolcu ve
# mürettebattan 1502'sinin ölümüyle sonuçlandı. Hayatta kalmanın bazı şans unsurları olsa da,
# bazı insan gruplarının hayatta kalma olasılığının diğerlerinden daha yüksek olduğu görülüyor.
# Bu projede ise yolcuların ödediği bilet ücretine odaklanacağız. Yolcu verilerini (sınıf, yaş,
# cinsiyet, aile büyüklüğü ve hayatta kalma durumu) kullanarak "bir yolcunun bilet ücretini
# hangi özellikleri belirliyor ve bu ücret ne kadar iyi tahmin edilebilir?" sorusuna yanıt arayacağız.

# Veriye Hızlıca göz atalım
print(data_giris.head().T)
print(data_giris.describe().T)
print("Boyut:", data_giris.shape)
print(data_giris.dtypes.T)
# Verimizde 891 satır yani gözlem ve 12 sutun yani değişken bulunuyor. Bazıları açıklayıcı
# olmamakla beraber 5 kategorik 7 sayısal değişken bulunuyor.
# Fare (bilet ücreti, ekonomik statünün bir göstergesi) bağımlı değişkeni, diğer faktörler ise
# bağımsız değişkenleri temsil etmektedir. Yolcu sınıfı, yaş, cinsiyet, aile büyüklüğü ve
# hayatta kalma durumunun bilet ücretini ne kadar açıkladığını farklı regresyon modelleriyle ölçeceğiz.

# Yolcu ID, İsim gibi açıklayıcı olmayan sütunların elenmesi.
data = data_giris.drop(["PassengerId", "Name", "Ticket", "Cabin", "Embarked"], axis=1)

# Kategorik Değişkeni Sayısallaştırma (Encoding) / Sex için 0-1 dönüşümü
data = pd.get_dummies(data, columns=["Sex"], drop_first=True)
# Bunu yapmaktaki amacımız bu proje kapsamında kuracağımız modellemelerde
# cinsiyet gibi kategorik ama önemli bir değişkeni modele dahil edebilmek.

# Eksik Gözlemleri Kontrol Edelim
print("Eksik gözlem sayısı:", data.isnull().sum().sum())
# Toplam 177 adet eksik gözlemimiz var (hepsi Age değişkeninde). Bunları direkt silelim.
data = data.dropna()

# Tekrar Veriye Bir Göz Atalım
print(data.head().T)
print(data.describe().T)
print("Boyut:", data.shape)
# Veriyi düzenledik, Ön işleme kısmını tamamladık şimdi modellemelere geçiş yapabiliriz.

# İlk Olarak Korelasyon Kontrolü
corr = data.corr()
mt.figure()
sb.heatmap(corr, annot=True)
grafik_kaydet("01_korelasyon")
# Bağımsız değişkenlerin birbiriyle bağlantısı oldukça düşük görünüyor. Fazla multicollinearity
# olmadığını aslında buradan görebiliriz. Şimdi modellerimizi kurarak başlayalım.


# %% 2. LİNEER REGRESYON MODELLERİ
# 2.1 Basit Doğrusal Regresyon
# Bu aşamada, bilet ücretini (Fare) tahmin etmek için sadece en güçlü olduğunu düşündüğümüz
# Pclass (Yolcu Sınıfı) değişkenini kullanacağız.
# Veriyi hazırlayalım
x = data[["Pclass"]]
y = data["Fare"]
# Sabit Terimi (Intercept) Ekleme
# statsmodels denklemde Beta_0'ı (sabit terimi) otomatik eklemez, bu yüzden biz ekleyelim.
x = sm.add_constant(x)
# OLS Modelini Kurma ve Fit Etme
model_basit = sm.OLS(y, x).fit()
# İstatistiksel Özeti Yazdırma
print(model_basit.summary())
# Modeli yorumlamak gerekirse:
# Düşük bir R^2 değerine sahip yani seçmiş olduğumuz yolcu sınıfı değişkeni bilet ücretlerini
# açıklamada zayıf kalmış, yaklaşık %30 unu açıklayabiliyor (R^2 = 0.307).
# p değeri 0 çıkmış yani anlamlı bir model olduğu söylenebilir.
# Yolcu sınıfındaki bir birimlik artış bilet ücretini yaklaşık -35 birim düşürmüş.
# Son Olarak MSE ve R^2 Değerlerine Bakalım
mse_basit = mean_squared_error(y, model_basit.predict(x))
r2_basit = r2_score(y, model_basit.predict(x))
print(f"Basit regresyon  MSE: {mse_basit:.2f}   R^2: {r2_basit:.4f}")


# 2.2 Çoklu Doğrusal Regresyon
# İlk olarak genel bir model kuralım sonra eğitim test parçalanması yaparız.
x = data.drop("Fare", axis=1)
x = x.astype(float)
# burada daha önce kategorik yaptığımız değişkenleri sayısal niceliğe dönüştürmemiz gerekiyor
y = data["Fare"]
x = sm.add_constant(x)
model_coklu = sm.OLS(y, x).fit()
print(model_coklu.summary())
# Yorumlayalım:
# Basit Regresyon modelimizde R^2 değerimiz 0.307 iken burada 0.370 değerine (düzeltilmiş R^2:
# 0.306 -> 0.365) yükselmiş. Yükselen değer yine de düşük olsa da yeni eklenen değişkenlerimizin
# modelin açıklanabilirliğini artırdığını söyleyebiliriz. Katsayıların anlamlılığı da gayet olumlu
# görünüyor, örneğin aile genişledikçe fiyat artıyor (SibSp, Parch). Aynı zamanda Sex_male durumu
# fiyata olumsuz bir etkiye sahip (o dönemki şartlar) ama istatistiksel olarak anlamlı değil.
# F istatistik değeri p<0.05 olduğundan değişkenlerden en az biri hedef değişken üzerinde anlamlı
# bir etkiye sahiptir deriz. Yine değişkenlerin p değerlerine baktığımızda Survived, Age ve
# Sex_male hariç diğer değişkenlerin anlamlı etkiye (p<0.05) sahip olduğunu söyleyebiliriz.
# Geri kalan anlamsız değişkenleri elemek için Lasso Regresyonu kullanmak gerekebilir.

# 2.2.1 Varsayım Kontrollerini İnceleyelim
# Normallik Grafik
residuals = model_coklu.resid
mt.figure()
stats.probplot(residuals, dist="norm", plot=mt)
grafik_kaydet("02_normallik_qq")
# Grafiğe baktığımızda çoğunluk normale yakın olsa da özellikle uç noktalarda ve bazı ara
# değerlerde çok ciddi sapmalar görünüyor ve bu da normalliğe pek yakın olmadığını gösteriyor.
# Test ile de bakalım
print("Shapiro-Wilk:", shapiro(residuals))
# Değer çok düşük, H0 ret yani veri normal dağılmıyor.
# Summary de Jarque-Bera değerinin yüksek olması ve Prob(JB)'nin 0 olması,
# hataların normal dağılmadığını gösterir ki bu da çıktımızı bulgumuzu doğrular.

# Otokorelasyon
mt.figure()
mt.scatter(range(len(residuals)), residuals)
mt.title("Otokorelasyon Kontrolü")
mt.xlabel("Gözlem")
mt.ylabel("Residuals")
mt.axhline(y=0)
grafik_kaydet("03_otokorelasyon")
# Yorumlarsak 0 çizgisi etrafında belli bir kurala bağlı olmayan rastgele bir dağılım var.
# Yani değişkenler arasında otokorelasyon yok diyebiliriz.
# Summary de Durbin-Watson testinde Değerin 2'ye yakın olması (2.02), hatalar arasında
# otokorelasyon olmadığını gösterir ki bu da regresyon çıktımızı doğrular.

# Değişen Varyans Kontrolü (Summary de gözüken bir değer değil)
mt.figure()
mt.scatter(model_coklu.fittedvalues, residuals)
mt.ylabel("Residuals")
mt.xlabel("Fitted")
mt.title("Sabit Varyans Kontrolü")
mt.axhline(y=0)
grafik_kaydet("04_sabit_varyans")
# Noktaların rastgele dağılmasını bekledik fakat kapanan bir huniye benzer bir şekil
# oluştuğunu görüyoruz. Buradan değişen varyans problemi var deriz.

# Multicollinearity (Çoklu Bağlantı) Kontrolü
X = data.drop("Fare", axis=1)
X = add_constant(X).astype(float)
# Her değişken için VIF hesapla
vif_data = pd.DataFrame()
vif_data["Değişken"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]
print(vif_data[vif_data["Değişken"] != "const"].sort_values(by="VIF", ascending=False))
# Bağımsız değişkenlerin birbiriyle zayıf ilişkide olmasını ve modelimizin anlamlı olmasını
# istiyoruz. VIF değeri 5 ten küçük olsun istiyoruz, 10'un üzeri ise ciddi bir sorun vardır diye
# yorumlarız. Tabloyu incelediğimizde tüm değişkenlerin VIF değerleri 1-2 civarında olduğunu
# görüyoruz. Yani modelimizde çoklu doğrusallık sorunu olmadığını anlıyoruz, değişkenlerimiz
# birbirinden bağımsız bilgi sunuyor.

# MSE ve R^2 Değerleri
mse_coklu = mean_squared_error(y, model_coklu.predict(x))
r2_coklu = r2_score(y, model_coklu.predict(x))
print(f"Çoklu regresyon  MSE: {mse_coklu:.2f}   R^2: {r2_coklu:.4f}")
# Basit doğrusal regresyona kıyasla daha iyi bir değerle karşılaştık.
# Hata değerimiz düşerken aynı zamanda R^2 değerimiz yükseldi.
# Yani değişken sayımız arttığında daha kapsamlı, daha düşük hatalı ve açıklayıcılığı yüksek bir
# model elde ettik. Fakat kontrol ettiğimiz varsayımların sağlanmadığı durumda biz bu modellerin
# çıktılarına tam anlamıyla güvenemeyiz, model ciddi oranda yanılabilir.
# Bu sorunları ortadan kaldırmak, mevcut varyans şişmesini baskılamak ve çoklu bağlantının
# yarattığı bu kararsızlığı dizginlemek için PCR gibi daha gelişmiş modellere geçiş yapılmalı.


# %% 2.2.2 Eğitim Test Parçalanması
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=SEED_BOLME)
print("Eğitim seti:", x_train.shape)   # 571 gözlem
print("Test seti  :", x_test.shape)    # 143 gözlem
# Şimdi eğitim modelimizi kuralım.
model_egitim = sm.OLS(y_train, x_train).fit()
print(model_egitim.summary())
# Şimdi test seti üzerinden tahmin oluşturup buradan MSE ve R^2 değerlerini kontrol edelim.
y_tahmin = model_egitim.predict(x_test)
mse_test = mean_squared_error(y_test, y_tahmin)
r2_test = r2_score(y_test, y_tahmin)
print(f"OLS test  MSE: {mse_test:.2f}   RMSE: {np.sqrt(mse_test):.2f}   R^2: {r2_test:.4f}")
# R^2 değerinin test setinde (0.449 -> 0.180) yaklaşık yarı yarıya bu kadar sert düşmesi, modelin
# eğitim verisine fazla uyduğunu ama yeni gördüğü test seti verilerine genelleyemediğini gösterir.
# Hata miktarının test setinde neredeyse iki katına çıkması, tahminlerinin test verisinde çok daha
# fazla saptığını gösteriyor. Genelde bu iki değerin birbirine yakın olması beklenir.
# Sonuçlar pek de parlak görünmüyor.
