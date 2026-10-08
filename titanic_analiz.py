# -*- coding: utf-8 -*-
"""
İSTATİSTİKSEL YAZILIMLAR PROJE ÖDEVİ (temiz sürüm)
Mehmet Hamza Şükür

Titanic yolcu verisinde bilet ücretini (Fare) tahmin eden regresyon modelleri:
basit/çoklu OLS, varsayım kontrolleri, VIF, eğitim-test ayrımı, PCR, PLS,
Ridge ve Lasso.

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
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso, LassoCV, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
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


# %% 2.3 PCR (PRINCIPAL COMPONENT REGRESSION)
# İlk önce cross validation yapısını kuralım
cv = KFold(n_splits=5, shuffle=True, random_state=SEED_CV)
# Modelimizin x ve y'sini tekrar yazalım ve eğitim test parçalanmasını yapalım
x = data.drop("Fare", axis=1)
y = data["Fare"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=SEED_BOLME)
# Şimdi veriyi scale edelim (boyut indirgemesi için)
sc = StandardScaler()
x_train_std = sc.fit_transform(x_train)
x_test_std = sc.transform(x_test)

# PCA'yı hazırlayalım ve Optimal Bileşen belirleyelim
pca = PCA()
x_train_pca = pca.fit_transform(x_train_std)
regr = LinearRegression()
bilesen_sayilari = np.arange(1, x_train_std.shape[1] + 1)
mse = []
for i in bilesen_sayilari:
    score = -1 * cross_val_score(regr, x_train_pca[:, :i], y_train,
                                 cv=cv, scoring="neg_mean_squared_error").mean()
    mse.append(score)
# Hangi bileşen en iyi mse değerini vermiş diye grafik ile bakalım.
mt.figure()
mt.plot(bilesen_sayilari, mse, marker="o")
mt.xlabel("Component Sayısı")
mt.ylabel("MSE")
mt.title("Bileşen Sayısına Göre Hata Değişimi (CV ile)")
grafik_kaydet("05_pcr_bilesen_secimi")
# En iyi bileşen sayısı, CV hatasının en küçük olduğu nokta olarak otomatik seçilir.
best_n = int(bilesen_sayilari[np.argmin(mse)])
print("PCR CV MSE:", [round(float(m), 1) for m in mse])
print("PCR için seçilen bileşen sayısı:", best_n, f"(toplam değişken sayısı: {x_train_std.shape[1]})")

# Şimdi Modelimizi tekrar kuralım.
pca_final = PCA(n_components=best_n)
x_train_final = pca_final.fit_transform(x_train_std)
x_test_final = pca_final.transform(x_test_std)
pcreg = LinearRegression().fit(x_train_final, y_train)
# Tahminleri alalım
y_test_pred = pcreg.predict(x_test_final)
# Performans Ölçümü
pcr_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
pcr_r2 = r2_score(y_test, y_test_pred)
pcr_mae = mean_absolute_error(y_test, y_test_pred)
print(f"PCR Model RMSE: {pcr_rmse:.4f}")
print(f"PCR Model R^2 Skoru: {pcr_r2:.4f}")
print(f"PCR Model MAE: {pcr_mae:.4f}")

# Modelin ortalama hatası (MAE) 25.42 birimdir. Ancak RMSE değerinin 58.44 gibi çok daha
# yüksek bir seviyede olması, veri setindeki aykırı değerlerin (uç fiyatlı biletlerin) model
# tahminlerini ciddi şekilde saptırdığını gösteriyor. %18'lik düşük R-kare skoru ile
# birleştiğinde; PCR modelinin bilet fiyatlarındaki genel eğilimi yakaladığı, fakat yüksek
# fiyatlı biletlerin yarattığı varyansı açıklamakta yetersiz kaldığını görüyoruz.
# ÖNEMLİ: Çapraz doğrulama tüm bileşenleri (6/6) seçtiği için bu PCR modeli boyut indirgemesi
# yapmıyor; sonuçları düz OLS'nin test sonuçlarıyla birebir aynıdır (R^2 = 0.1798).


# %% 2.4 PLSR (PARTIAL LEAST SQUARES REGRESSION)
# Öncelikle modelimizin x ve y eğitim test parçalanması yukarıdaki ile aynıdır.
cv = KFold(n_splits=5, shuffle=True, random_state=SEED_CV)

# Optimal Bileşen Sayısını Belirleme
pls_bilesenler = np.arange(1, x_train.shape[1] + 1)
pls_mse = []
for i in pls_bilesenler:
    pls = PLSRegression(n_components=i)
    score = -1 * cross_val_score(pls, x_train, y_train,
                                 cv=cv, scoring="neg_mean_squared_error").mean()
    pls_mse.append(score)
# Grafik ile de kontrol edelim.
mt.figure()
mt.plot(pls_bilesenler, pls_mse, marker="o")
mt.xlabel("Bileşen Sayısı (PLS)")
mt.ylabel("MSE")
mt.title("PLS: Bileşen Sayısına Göre Hata Değişimi (CV ile)")
grafik_kaydet("06_pls_bilesen_secimi")

print("PLS CV MSE:", [round(float(m), 1) for m in pls_mse])
# Grafiğe baktığımızda en uygun bileşen sayısının 2 olduğunu görüyoruz. Bu değerle model kuralım.
best_pls = 2
pls_model = PLSRegression(n_components=best_pls).fit(x_train, y_train)
y_test_pred = pls_model.predict(x_test)
# Metrikler
pls_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
pls_r2 = r2_score(y_test, y_test_pred)
pls_mae = mean_absolute_error(y_test, y_test_pred)
print("--- PLS Final Model Sonuçları ---")
print(f"PLS RMSE: {pls_rmse:.4f}")
print(f"PLS R^2 : {pls_r2:.4f}")
print(f"PLS MAE : {pls_mae:.4f}")

# Genel beklentinin aksine, bu veri seti üzerinde PCR modelinin PLS modelinden hafifçe daha iyi
# performans gösterdiğini gözlemledik. Bu durum, bağımsız değişkenlerin hedef değişken üzerindeki
# etkisinin düşük olması veya aykırı değerlerin PLS bileşenlerini gürültülü hale getirmesinden
# kaynaklanıyor olabilir. Bu iki model de (PCR ve PLS) bilet fiyatındaki değişkenliğin ancak
# %17-%18'ini açıklayabildi. RMSE hala çok yüksek (58 birim).
# Not: PCR'da CV tüm bileşenleri (6/6) seçtiği için PCR sonucu düz OLS ile aynıdır. PLS'te CV
# hatası 2 bileşenden sonra neredeyse düz kaldığı için (1426 -> 1413) en sade model olan 2
# bileşen tercih edildi.


# %% 2.5 RİDGE REGRESYONU
# \hat(\beta) = (X^T X + \lambda I)^{-1} X^T Y
# Ridge ve Lasso değişkenlerin ölçeğine duyarlı olduğu için PCR'da hazırladığımız
# standartlaştırılmış veriyi (x_train_std, x_test_std) kullanalım.
# Önce lambda (alpha) için bir aday listesi belirleyelim ve her adayı PCR ve PLS'te
# kurduğumuz aynı yapıyla 5 katlı cross validation ile deneyelim.
# Hatayı eğitim verisi üzerinde değil, modelin görmediği katlarda ölçüyoruz; çünkü eğitim
# hatasına bakarsak her zaman en küçük alpha kazanır.
cv = KFold(n_splits=5, shuffle=True, random_state=SEED_CV)
alpha_cand = np.array([0.0001, 0.001, 0.01, 0.1, 0.2, 0.5, 1, 5, 10, 20])
ridge_mse = []
for a in alpha_cand:
    skor = -1 * cross_val_score(Ridge(alpha=a), x_train_std, y_train,
                                cv=cv, scoring="neg_mean_squared_error").mean()
    ridge_mse.append(skor)
mt.figure()
mt.plot(alpha_cand, ridge_mse, marker="o")
mt.xscale("log")
mt.xlabel("alpha (lambda)")
mt.ylabel("CV MSE")
mt.title("Ridge: Alpha'ya Göre Hata Değişimi (CV ile)")
grafik_kaydet("07_ridge_alpha")

best_alpha_ridge = float(alpha_cand[np.argmin(ridge_mse)])
print("Ridge CV MSE:", [round(float(m), 1) for m in ridge_mse])
print("Ridge için seçilen alpha:", best_alpha_ridge)
ridge_model = Ridge(alpha=best_alpha_ridge).fit(x_train_std, y_train)
predicts_ridge = ridge_model.predict(x_test_std)
ridge_rmse = np.sqrt(mean_squared_error(y_test, predicts_ridge))
ridge_r2 = r2_score(y_test, predicts_ridge)
ridge_mae = mean_absolute_error(y_test, predicts_ridge)
print("--- Ridge Final Model Sonuçları ---")
print(f"Ridge RMSE: {ridge_rmse:.4f}")
print(f"Ridge R^2 : {ridge_r2:.4f}")
print(f"Ridge MAE : {ridge_mae:.4f}")
# Ridge'in CV hatası denenen tüm alpha değerlerinde neredeyse aynı (1413.5 -> 1412.5); en küçük
# hata alpha = 10'da. Test setinde RMSE 58.40, R^2 0.1807 ile OLS'ye (58.44 / 0.1798) çok yakın,
# MAE'de ise biraz daha iyi (25.15 - 25.42). Çoklu bağlantı sorunu olmadığı için (VIF 1-2 civarı)
# Ridge'in cezalandırması bu veri setinde belirgin bir fark yaratmıyor, beklenen bir sonuç.


# %% 2.6 LASSO REGRESYONU
# alpha optimizasyonunu LassoCV ile (10 katlı cross validation) yapalım.
lasso_cv = LassoCV(cv=10, random_state=34, max_iter=1000)
lasso_cv.fit(x_train_std, y_train)
print("Lasso için seçilen alpha:", lasso_cv.alpha_)

lasso_model = Lasso(alpha=lasso_cv.alpha_).fit(x_train_std, y_train)
predicts_lasso = lasso_model.predict(x_test_std)
lasso_rmse = np.sqrt(mean_squared_error(y_test, predicts_lasso))
lasso_r2 = r2_score(y_test, predicts_lasso)
lasso_mae = mean_absolute_error(y_test, predicts_lasso)
print("--- Lasso Final Model Sonuçları ---")
print(f"Lasso RMSE: {lasso_rmse:.4f}")
print(f"Lasso R^2 : {lasso_r2:.4f}")
print(f"Lasso MAE : {lasso_mae:.4f}")

# Lasso hangi değişkenlerin katsayısını sıfırladı?
lasso_katsayi = pd.Series(lasso_model.coef_, index=x_train.columns)
print("Lasso katsayıları (standartlaştırılmış veri):\n", lasso_katsayi.round(3))
print("Sıfırlanan değişkenler:", list(lasso_katsayi[lasso_katsayi == 0].index))
# Lasso (alpha = 0.148) hiçbir değişkenin katsayısını sıfırlamadı; yani anlamsız çıkan Survived,
# Age ve Sex_male değişkenlerini de modelde tuttu (katsayıları küçük: 1.2, -1.8, -1.4). Çoklu
# regresyonda elemeyi düşündüğümüz bu değişkenler için Lasso bir eleme yapmadı, sadece
# katsayılarını hafifçe küçülttü. Test performansı OLS ve Ridge ile neredeyse aynı (RMSE 58.41,
# R^2 0.1804). Pclass (-30.0) fiyatı en çok etkileyen değişken olmaya devam ediyor.


# %% 3. TEST SETİ SONUÇ ÖZETİ
ozet = pd.DataFrame(
    {
        "RMSE": [np.sqrt(mse_test), pcr_rmse, pls_rmse, ridge_rmse, lasso_rmse],
        "MAE": [mean_absolute_error(y_test, y_tahmin), pcr_mae, pls_mae, ridge_mae, lasso_mae],
        "R2": [r2_test, pcr_r2, pls_r2, ridge_r2, lasso_r2],
    },
    index=["OLS", f"PCR ({best_n} bileşen)", f"PLS ({best_pls} bileşen)",
           f"Ridge (alpha={best_alpha_ridge})", f"Lasso (alpha={lasso_cv.alpha_:.4f})"],
).round(4)
print("\nTest seti karşılaştırması:\n", ozet)

# Devamı (KNN, SVR, yapay sinir ağı ve ağaç tabanlı modeller) bir sonraki aşamada eklenecek.
