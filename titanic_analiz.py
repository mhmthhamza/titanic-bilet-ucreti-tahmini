# -*- coding: utf-8 -*-
"""
İSTATİSTİKSEL YAZILIMLAR PROJE ÖDEVİ (temiz sürüm)
Mehmet Hamza Şükür

Titanic yolcu verisinde bilet ücretini (Fare) tahmin eden regresyon modelleri:
basit/çoklu OLS, varsayım kontrolleri, VIF, eğitim-test ayrımı, PCR, PLS,
Ridge, Lasso, KNN, SVR, yapay sinir ağı, CART, Bagging, Random Forest ve
Gradient Boosting.

Kullanım: Spyder'da hücre hücre (Shift+Enter) ya da `python titanic_analiz.py`
Girdi   : veri/Titanic-Dataset.csv
Çıktı   : grafikler/*.png
"""

# %% 0. Kütüphaneler ve ayarlar
import warnings
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
from sklearn.ensemble import BaggingRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
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


# %% 2.7 K-EN YAKIN KOMŞU (KNN) REGRESYONU
# Doğrusal olmayan regresyon modellerine geçelim. Önce varsayılan (k = 5) modeli kuralım.
knn_model = KNeighborsRegressor().fit(x_train, y_train)
knn_pred = knn_model.predict(x_test)
print(f"KNN (k={knn_model.n_neighbors}) test R^2: {r2_score(y_test, knn_pred):.4f}")

# Optimum k değerini GridSearchCV ile belirleyelim.
k_params = {"n_neighbors": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40]}
knn_grid = GridSearchCV(KNeighborsRegressor(), k_params, cv=5).fit(x_train, y_train)
print("KNN için seçilen k:", knn_grid.best_params_)

knn_model = KNeighborsRegressor(n_neighbors=knn_grid.best_params_["n_neighbors"]).fit(x_train, y_train)
knn_pred = knn_model.predict(x_test)
knn_rmse = np.sqrt(mean_squared_error(y_test, knn_pred))
knn_r2 = r2_score(y_test, knn_pred)
knn_mae = mean_absolute_error(y_test, knn_pred)
print("--- KNN Final Model Sonuçları ---")
print(f"KNN RMSE: {knn_rmse:.4f}")
print(f"KNN R^2 : {knn_r2:.4f}")
print(f"KNN MAE : {knn_mae:.4f}")
# Varsayılan k = 5 ile test R^2 yalnızca 0.086 çıkmıştı; GridSearchCV k = 9'u seçti ve R^2 0.154'e
# yükseldi. KNN, RMSE'de (59.35) doğrusal modellerden biraz kötü, ama MAE'de (19.80) belirgin
# şekilde daha iyi (doğrusal modellerde ~25). Yani KNN ortadaki yaygın bilet fiyatlarını daha iyi
# tahmin ediyor, uçtaki pahalı biletlerde ise yine büyük hata yapıyor.


# %% 2.8 DESTEK VEKTÖR MAKİNELERİ (SVR) REGRESYONU
# Doğrusal SVR
svr_lin = SVR(kernel="linear").fit(x_train, y_train)
svr_lin_pred = svr_lin.predict(x_test)
print(f"SVR doğrusal (varsayılan) test R^2: {r2_score(y_test, svr_lin_pred):.4f}")

# Doğrusal olmayan SVR
svr_rbf = SVR(kernel="rbf").fit(x_train, y_train)
svr_rbf_pred = svr_rbf.predict(x_test)
svr_rbf_rmse = np.sqrt(mean_squared_error(y_test, svr_rbf_pred))
svr_rbf_r2 = r2_score(y_test, svr_rbf_pred)
svr_rbf_mae = mean_absolute_error(y_test, svr_rbf_pred)
print("--- SVR (rbf, varsayılan) Sonuçları ---")
print(f"SVR rbf RMSE: {svr_rbf_rmse:.4f}")
print(f"SVR rbf R^2 : {svr_rbf_r2:.4f}")
print(f"SVR rbf MAE : {svr_rbf_mae:.4f}")

# En iyi doğrusal modeli seçmek için C parametresini GridSearchCV ile arayalım.
params_svr = {"C": np.arange(0.1, 4, 0.1)}
svr_grid = GridSearchCV(SVR(kernel="linear"), params_svr, cv=5).fit(x_train, y_train)
best_C = float(svr_grid.best_params_["C"])
print("SVR (doğrusal) için seçilen C:", round(best_C, 1))

svr_model = SVR(kernel="linear", C=best_C).fit(x_train, y_train)
svr_pred = svr_model.predict(x_test)
svr_rmse = np.sqrt(mean_squared_error(y_test, svr_pred))
svr_r2 = r2_score(y_test, svr_pred)
svr_mae = mean_absolute_error(y_test, svr_pred)
print("--- SVR (doğrusal, ayarlı) Sonuçları ---")
print(f"SVR doğrusal RMSE: {svr_rmse:.4f}")
print(f"SVR doğrusal R^2 : {svr_r2:.4f}")
print(f"SVR doğrusal MAE : {svr_mae:.4f}")
# Varsayılan rbf çekirdekli SVR bu veride çok kötü çıktı (test R^2 = -0.055, yani ortalamayı
# tahmin etmekten bile kötü). Doğrusal SVR ise C = 3.8 ile R^2 0.145'e ulaştı. Doğrusal SVR'ın MAE'si
# (17.80) doğrusal regresyon modellerinden (~25) çok daha iyi, RMSE'si (59.67) ise biraz kötü;
# SVR ortadaki fiyatları daha iyi, uçtaki pahalı biletleri daha kötü tahmin ediyor.


# %% 2.9 YAPAY SİNİR AĞLARI (ANN) (MLP)
# Yapay sinir ağları modellemede standartlaştırılmış girdiler kullanılır.
scl = StandardScaler()
scl.fit(x_train)
x_train_scl = scl.transform(x_train)
x_test_scl = scl.transform(x_test)

ann_model = MLPRegressor()
# Grid search için parametre aralıkları
ann_params = {"alpha": [0.005, 0.01, 0.05, 0.1],
              "hidden_layer_sizes": [(20, 20), (100, 150, 200),
                                     (300, 200, 100),
                                     (8, 8, 8), (8, 8)],
              "activation": ["relu", "logistic"]}
# Büyük ağlar bazen belirlenen iterasyonda yakınsamayabilir (ConvergenceWarning),
# bu uyarıları gizleyip sonuçlara bakalım.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=ConvergenceWarning)
    ann_grid = GridSearchCV(ann_model, ann_params, cv=5).fit(x_train_scl, y_train)
print("YSA için seçilen parametreler:", ann_grid.best_params_)

with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=ConvergenceWarning)
    ann_model = MLPRegressor(**ann_grid.best_params_).fit(x_train_scl, y_train)
ann_pred = ann_model.predict(x_test_scl)
ann_rmse = np.sqrt(mean_squared_error(y_test, ann_pred))
ann_r2 = r2_score(y_test, ann_pred)
ann_mae = mean_absolute_error(y_test, ann_pred)
print("--- YSA Final Model Sonuçları ---")
print(f"YSA RMSE: {ann_rmse:.4f}")
print(f"YSA R^2 : {ann_r2:.4f}")
print(f"YSA MAE : {ann_mae:.4f}")
# Bir çalıştırmada GridSearchCV relu, alpha = 0.05 ve (100, 150, 200) yapısını seçti; test setinde
# RMSE 56.87, R^2 0.223, MAE 16.73 elde edildi. Başka bir çalıştırmada ise R^2 0.189'a düştü.
# YSA başlangıçta rastgele ağırlıklarla başladığı için (random_state verilmedi) kodu her
# çalıştırdığımızda sonuçlar ve seçilen parametreler değişebiliyor; bu yüzden YSA'yı kesin bir
# sıralamayla değil, yaklaşık bir aralıkla (R^2 ≈ 0.19-0.23, MAE ≈ 17-18) yorumlamak gerekir.


# %% 2.10 CART (CLASSIFICATION AND REGRESSION TREES) - KARAR AĞACI
# Önce varsayılan modeli kuralım.
cart_model = DecisionTreeRegressor().fit(x_train, y_train)
pred_cart = cart_model.predict(x_test)
print(f"CART (varsayılan) test R^2: {r2_score(y_test, pred_cart):.4f}")

# Model Optimizasyonu
cart_params = {
    "min_samples_split": range(2, 100),
    "max_leaf_nodes": range(2, 10)}
grid_cart_model = GridSearchCV(cart_model, cart_params, cv=10).fit(x_train, y_train)
print("CART için seçilen parametreler:", grid_cart_model.best_params_)

# Bulunan en iyi parametrelerle modeli yeniden kuralım.
optim_cart_model = DecisionTreeRegressor(**grid_cart_model.best_params_).fit(x_train, y_train)
best_pred_cart = optim_cart_model.predict(x_test)
cart_rmse = np.sqrt(mean_squared_error(y_test, best_pred_cart))
cart_r2 = r2_score(y_test, best_pred_cart)
cart_mae = mean_absolute_error(y_test, best_pred_cart)
print("--- CART Final Model Sonuçları ---")
print(f"CART RMSE: {cart_rmse:.4f}")
print(f"CART R^2 : {cart_r2:.4f}")
print(f"CART MAE : {cart_mae:.4f}")
# Varsayılan ağaçla test R^2 0.153 idi; GridSearchCV (max_leaf_nodes = 8, min_samples_split = 36)
# ile 0.184'e yükseldi. Budanmış ağaç RMSE'de (58.28) doğrusal modellerle yaklaşık aynı, MAE'de
# (19.07) ise onlardan belirgin şekilde daha iyi.


# %% 2.11 BOOTSTRAP AGGREGATION (BAGGING)
bagging_model = BaggingRegressor(bootstrap_features=True).fit(x_train, y_train)
bag_preds = bagging_model.predict(x_test)
print(f"Bagging (varsayılan) test R^2: {r2_score(y_test, bag_preds):.4f}")

# Model Optimizasyonu
bag_params = {"n_estimators": range(2, 40)}
grid_bag_model = GridSearchCV(bagging_model, bag_params, cv=10).fit(x_train, y_train)
print("Bagging için seçilen parametreler:", grid_bag_model.best_params_)

best_bag_model = BaggingRegressor(bootstrap_features=True,
                                  **grid_bag_model.best_params_).fit(x_train, y_train)
best_bag_preds = best_bag_model.predict(x_test)
bag_rmse = np.sqrt(mean_squared_error(y_test, best_bag_preds))
bag_r2 = r2_score(y_test, best_bag_preds)
bag_mae = mean_absolute_error(y_test, best_bag_preds)
print("--- Bagging Final Model Sonuçları ---")
print(f"Bagging RMSE: {bag_rmse:.4f}")
print(f"Bagging R^2 : {bag_r2:.4f}")
print(f"Bagging MAE : {bag_mae:.4f}")
# Bir çalıştırmada varsayılan Bagging ile test R^2 0.149 idi; GridSearchCV ile ağaç sayısı 36
# seçilince 0.205'e çıktı (RMSE 57.53, MAE 18.65). Başka bir çalıştırmada ise R^2 0.156 civarında
# kaldı. Bagging'de ağaçlar rastgele örneklemlerle kurulduğu için (random_state verilmedi) sonuç
# çalıştırmadan çalıştırmaya belirgin şekilde değişiyor; bazen tek bir ağaçtan (CART) daha iyi,
# bazen ona yakın çıkıyor.


# %% 2.12 RANDOM FOREST (RF)
rf_model = RandomForestRegressor().fit(x_train, y_train)
pred_rf = rf_model.predict(x_test)
print(f"Random Forest (varsayılan) test R^2: {r2_score(y_test, pred_rf):.4f}")

# Model Optimizasyonu
# Veri setimizde 6 değişken olduğu için max_features adayları [2, 3, 5] ile sınırlandı.
# n_jobs=-1: ızgara çok uzun sürdüğü için işlemcinin tüm çekirdekleri kullanılır (sonucu değiştirmez).
rf_params = {"max_depth": range(1, 15), "max_features": [2, 3, 5], "n_estimators": [100, 250, 500, 1000]}
rf_grid_model = GridSearchCV(rf_model, rf_params, cv=10, n_jobs=-1).fit(x_train, y_train)
print("Random Forest için seçilen parametreler:", rf_grid_model.best_params_)

rf_best_model = RandomForestRegressor(**rf_grid_model.best_params_).fit(x_train, y_train)
pred_rf_best = rf_best_model.predict(x_test)
rf_rmse = np.sqrt(mean_squared_error(y_test, pred_rf_best))
rf_r2 = r2_score(y_test, pred_rf_best)
rf_mae = mean_absolute_error(y_test, pred_rf_best)
print("--- Random Forest Final Model Sonuçları ---")
print(f"Random Forest RMSE: {rf_rmse:.4f}")
print(f"Random Forest R^2 : {rf_r2:.4f}")
print(f"Random Forest MAE : {rf_mae:.4f}")
# Bir çalıştırmada GridSearchCV max_depth = 5, max_features = 3, n_estimators = 500 değerlerini
# seçti; test setinde RMSE 57.27, R^2 0.212, MAE 17.18 elde edildi (varsayılan Random Forest:
# R^2 0.158). Başka bir çalıştırmada R^2 0.202, MAE 17.34 çıktı. Random Forest'ın sonuçları
# rastgelelik nedeniyle biraz değişse de, iki çalıştırmada da en düşük MAE değerleri arasında yer aldı.


# %% 2.13 GRADIENT BOOSTING
gb_model = GradientBoostingRegressor().fit(x_train, y_train)
pred_gb = gb_model.predict(x_test)
print(f"Gradient Boosting (varsayılan) test R^2: {r2_score(y_test, pred_gb):.4f}")

# Model Optimizasyonu
gb_params = {"learning_rate": [0.01, 0.1], "max_depth": [3, 5, 10],
             "n_estimators": [100, 250, 500], "subsample": [1, 0.5]}
gb_grid = GridSearchCV(gb_model, gb_params, cv=5).fit(x_train, y_train)
print("Gradient Boosting için seçilen parametreler:", gb_grid.best_params_)

gb_best_model = GradientBoostingRegressor(**gb_grid.best_params_).fit(x_train, y_train)
preds_gb_best = gb_best_model.predict(x_test)
gb_rmse = np.sqrt(mean_squared_error(y_test, preds_gb_best))
gb_r2 = r2_score(y_test, preds_gb_best)
gb_mae = mean_absolute_error(y_test, preds_gb_best)
print("--- Gradient Boosting Final Model Sonuçları ---")
print(f"Gradient Boosting RMSE: {gb_rmse:.4f}")
print(f"Gradient Boosting R^2 : {gb_r2:.4f}")
print(f"Gradient Boosting MAE : {gb_mae:.4f}")
# Bir çalıştırmada varsayılan Gradient Boosting test R^2 = 0.201 ile iyi bir sonuç verdi; GridSearchCV
# (learning_rate = 0.01, max_depth = 3, n_estimators = 250, subsample = 0.5) ile ayarlanan model ise
# test setinde 0.192 çıktı (RMSE 58.01, MAE 17.95). Başka bir çalıştırmada ayarlı model R^2 0.208
# ile tablodaki en yüksek R^2'yi verdi. Veri küçük olduğu için bu farklar şansa bağlı olabilir.
# Not: YSA, Bagging, Random Forest ve Gradient Boosting'de rastgelelik var (random_state verilmedi);
# bu dört modelin sonuçları ve aralarındaki sıralama her çalıştırmada değişebiliyor. Kararlı olan
# bulgu şu: doğrusal olmayan modellerin MAE değeri (≈17-20) doğrusal modellerden (≈25) belirgin
# şekilde düşük.


# %% 3. TEST SETİ SONUÇ ÖZETİ
ozet = pd.DataFrame(
    {
        "RMSE": [np.sqrt(mse_test), pcr_rmse, pls_rmse, ridge_rmse, lasso_rmse, knn_rmse, svr_rmse,
                 svr_rbf_rmse, ann_rmse, cart_rmse, bag_rmse, rf_rmse, gb_rmse],
        "MAE": [mean_absolute_error(y_test, y_tahmin), pcr_mae, pls_mae, ridge_mae, lasso_mae,
                knn_mae, svr_mae, svr_rbf_mae, ann_mae, cart_mae, bag_mae, rf_mae, gb_mae],
        "R2": [r2_test, pcr_r2, pls_r2, ridge_r2, lasso_r2, knn_r2, svr_r2, svr_rbf_r2, ann_r2,
               cart_r2, bag_r2, rf_r2, gb_r2],
    },
    index=["OLS", f"PCR ({best_n} bileşen)", f"PLS ({best_pls} bileşen)",
           f"Ridge (alpha={best_alpha_ridge})", f"Lasso (alpha={lasso_cv.alpha_:.4f})",
           f"KNN (k={knn_grid.best_params_['n_neighbors']})", f"SVR doğrusal (C={best_C:.1f})",
           "SVR rbf", "YSA", "CART", "Bagging", "Random Forest", "Gradient Boosting"],
).round(4)
print("\nTest seti karşılaştırması:\n", ozet)
