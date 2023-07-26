import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import statsmodels.api as sm

from scipy.stats import skew,kurtosis,norm
from sklearn.metrics import mean_squared_error,mean_absolute_error
from sklearn.metrics import r2_score,confusion_matrix,accuracy_score,precision_score,recall_score,f1_score
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LinearRegression,Ridge,Lasso
from sklearn.neighbors import KNeighborsClassifier

#Ucitavanje baze
podaci=pd.read_csv('ShanghaiPM20100101_20151231.csv')

#Pregled podataka
print("Podaci:\n",podaci.head(10))

#Opis podataka
print("Format podataka:\n",podaci.shape)
print("Broj uzoraka:\n",podaci.shape[0])
print("Broj obelezja:\n",podaci.shape[1])
print("Tipovi obelezja:\n",podaci.dtypes)
print("Procenat nedostajucih podataka:\n",podaci.isna().sum()/podaci.shape[0]*100)
inicijalni_opis_podataka=podaci.describe()
print(inicijalni_opis_podataka)

#Zamena nevalidnih vrednosti za sate 
podaci['hour']=podaci['hour']+1

#Izbacivanje odgovarajucih obelezja
podaci.drop(['PM_Jingan','PM_Xuhui','No'], axis=1, inplace=True)

#Brisanje odredjenih redova sa nedostajucim podacima
podaci=podaci.dropna(subset=['DEWP', 'HUMI','PRES','TEMP','cbwd','Iws'])

#Brisanje uzoraka iz godine 2010
uzorci_2010=podaci[podaci['year']==2010].index
podaci.drop(uzorci_2010,inplace=True)

#Brisanje uzoraka iz godine 2011
uzorci_2011=podaci[podaci['year']==2011].index
podaci.drop(uzorci_2011,inplace=True)

#Zamena nedostajucih vrednosti 
podaci.fillna(method='bfill', inplace=True)
print("Procenat nedostajucih podataka:\n",podaci.isna().sum())
opis_podataka_nakon_sredjivanja_NaN_vrednosti=podaci.describe()

#Pretvaranje kategorickih obelezja u numericka
podaci_dummy = pd.get_dummies(podaci['cbwd'], prefix='SOF')
podaci = pd.concat([podaci, podaci_dummy], axis=1)
podaci.drop(['cbwd'], axis=1, inplace=True)
finalni_opis_podataka=podaci.describe()

#Analiza obelezja(statisticke velicine, raspodela)
#Raspodela temperature po godinama
podaci_year = podaci.set_index('year')
plt.hist(podaci_year.loc[2012,'TEMP'], density=True, alpha=0.5, bins=50, label = '2012')
plt.hist(podaci_year.loc[2013,'TEMP'], density=True, alpha=0.5, bins=50, label='2013')
plt.xlabel('Temperatura (℃)')
plt.ylabel('Verovatnoća')
plt.legend()
plt.show()

#Raspodela vlaznosti vazduha po godinama
plt.hist(podaci_year.loc[2014,'HUMI'], density=True, alpha=0.5, bins=50, label = '2014')
plt.hist(podaci_year.loc[2015,'HUMI'], density=True, alpha=0.5, bins=50, label='2015')
plt.xlabel('Vlaznost vazduha')
plt.ylabel('Verovatnoća')
plt.legend()
plt.show()


#Raspodela vazdusnog pritiska po sezonama
podaci_season = podaci.set_index('season')
plt.hist(podaci_season.loc[1,'PRES'], density=True, alpha=0.5, bins=50, label = '1')
plt.hist(podaci_season.loc[2,'PRES'], density=True, alpha=0.5, bins=50, label='2')
plt.xlabel('Vazdusni pritisak')
plt.ylabel('Verovatnoća')
plt.legend()
plt.show()

#Detaljna analiza vrednosti obelezja PM_US Post
#Statisticki opis obelezja
podaci_PM_US_POST=podaci['PM_US Post']
opsi_PM_US_POST=podaci_PM_US_POST.describe()
podaci_PM_US_POST.plot.box()




#Koeficijent asimetrije i spoljstenosti za 2013
print('Koeficijent asimetrije:  %.2f' % skew(podaci_year.loc[2013,'PM_US Post']))
print('Koeficijent spljostenosti:  %.2f' % kurtosis(podaci_year.loc[2013,'PM_US Post']))
PM_US_POST_2013 = podaci_year.loc[2013,'PM_US Post']
sb.distplot(PM_US_POST_2013, fit=norm)
plt.xlabel('Koncentracija cestica')
plt.ylabel('Verovatnoća')
plt.show()

#Prosecna koncentracija cestica po godinama
PM_US_POST_PROSEK_PO_GODINAMA=podaci[['year','month','PM_US Post']]
gb_po_godinama = PM_US_POST_PROSEK_PO_GODINAMA.groupby(by=['year', 'month']).mean()
T_G2012 = gb_po_godinama.loc[2012]['PM_US Post']
T_G2013 = gb_po_godinama.loc[2013]['PM_US Post']
T_G2014 = gb_po_godinama.loc[2014]['PM_US Post']
T_G2015 = gb_po_godinama.loc[2015]['PM_US Post']
plt.plot(np.arange(1, 13, 1), T_G2012, 'b', label='2012') 
plt.plot(np.arange(1, 13, 1), T_G2013, 'r', label='2013')
plt.plot(np.arange(1, 13, 1), T_G2014, 'g', label='2014')
plt.plot(np.arange(1, 13, 1), T_G2015, 'y', label='2015')
plt.xlim(0, 13)
plt.ylabel('Prosečna koncentracija cestica')
plt.xlabel('Mesec')
plt.legend()
plt.show()

#Prosecna koncentracija cestica po godisnjim dobima
PM_US_POST_PROSEK_PO_GODISNJIM_DOBIMA=podaci[['season','PM_US Post']]
gb_po_godisnjim_dobima = PM_US_POST_PROSEK_PO_GODISNJIM_DOBIMA.groupby(by=['season']).mean()

#Korelacija prosecne koncentracije cestica u odnosu na godine
podaci_month = pd.DataFrame()
for i in podaci_year.index.unique():
    podaci_month[i] = gb_po_godinama.loc[i, 'PM_US Post']
c=podaci_month[2012].corr(podaci_month[2014])
print("Korelacija za 2012 i 2014: %.3f" % c)

#Matrica korelacije PM US Post i TEMP
pm_us_post_temp_korelacije = podaci.loc[:,['PM_US Post','TEMP']]
matrica_korelacije=pm_us_post_temp_korelacije.corr()
sb.heatmap(matrica_korelacije, annot=True)
plt.show()

#Matrica korelacije PM US Post i temperatura rose
pm_us_post_dewp_korelacije = podaci.loc[:,['PM_US Post','DEWP']]
matrica_korelacije=pm_us_post_dewp_korelacije.corr()
sb.heatmap(matrica_korelacije, annot=True)
plt.show()

#Matrica korelacije PM US Post i kumulativne brzine vetra
pm_us_post_iws_korelacije = podaci.loc[:,['PM_US Post','Iws']]
matrica_korelacije=pm_us_post_iws_korelacije.corr()
sb.heatmap(matrica_korelacije, annot=True)
plt.show()

#Matrica korelacije za 2013 godinu
matrica_korelacije = podaci_month.corr() 
print(matrica_korelacije[2013])


#LINEARNA REGRESIJA
#Mera uspesnosti regresora
def model_evaluation(y_test, y_predicted):
    mse = mean_squared_error(y_test, y_predicted) 
    mae = mean_absolute_error(y_test, y_predicted) 
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_predicted)
    print('Mean squared error: ', mse)
    print('Mean absolute error: ', mae)
    print('Root mean squared error: ', rmse)
    print('R2 score: ', r2)
    res=pd.concat([pd.DataFrame(y_test.values), pd.DataFrame(y_predicted)], axis=1)
    res.columns = ['y', 'y_pred']
    print(res.head(20))

#Podela uzoraka na trening, validacioni i test skup
x = podaci.drop(['PM_US Post'], axis=1).copy()
y = podaci['PM_US Post'].copy()
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.15,random_state=10)
X_train1,X_val,y_train1,y_val=train_test_split(X_train,y_train,test_size=0.15,random_state=10)

#Standardizacija obelezja
numeric_feats = [item for item in x.columns if 'SOF' not in item]
dummy_feats = [item for item in x.columns if 'SOF' in item]
scaler = StandardScaler()
scaler.fit(X_train1[numeric_feats])
x_train_std = pd.DataFrame(scaler.transform(X_train1[numeric_feats]), columns = numeric_feats)
x_test_std = pd.DataFrame(scaler.transform(X_val[numeric_feats]), columns = numeric_feats)
x_train_std = pd.concat([x_train_std, X_train1[dummy_feats].reset_index(drop=True)], axis=1)
x_test_std = pd.concat([x_test_std, X_val[dummy_feats].reset_index(drop=True)], axis=1)
print(x_train_std.head())

# Osnovni oblik linearne regresije sa hipotezom y=b0+b1x1+b2x2+...+bnxn
# Inicijalizacija
print("Osnovni oblik linearne regresije:\n")
first_regression_model = LinearRegression(fit_intercept=True)

# Obuka
first_regression_model.fit(X_train1, y_train1)

# Testiranje
y_predicted = first_regression_model.predict(X_val)

# Evaluacija
model_evaluation(y_val, y_predicted)

# Ilustracija koeficijenata
plt.figure(figsize=(10,5))
plt.bar(range(len(first_regression_model.coef_)),first_regression_model.coef_)
plt.show()
print("koeficijenti: ", first_regression_model.coef_)

#Obuka standardizovanog modela
print("Standardizovani oblik linearne regresije:\n")
std_regression_model = LinearRegression(fit_intercept=True)
std_regression_model.fit(x_train_std, y_train1)
y_predicted = std_regression_model.predict(x_test_std)
model_evaluation(y_val, y_predicted)
plt.figure(figsize=(10,5))
plt.bar(range(len(std_regression_model.coef_)),std_regression_model.coef_)
plt.show()
print("koeficijentie: ", std_regression_model.coef_)

#Selekcija obelezja
X = sm.add_constant(X_train1)
model = sm.OLS(y_train1, X.astype('float')).fit()
print(model.summary())

#Izbacivanje obelezja sa odgovarajucim p-vrednostima
print("Oblik linearne regresije sa izbacenim jednim obelezjem:\n")
X_train_11 = sm.add_constant(X_train1.drop('day', axis=1))
X_val_11=sm.add_constant(X_val.drop('day', axis=1))
model = sm.OLS(y_train1, X_train_11.astype('float')).fit()
print(model.summary())
second_regression_model = LinearRegression(fit_intercept=True)
second_regression_model.fit(X_train_11, y_train1)
y_predicted = second_regression_model.predict(X_val_11)
model_evaluation(y_val, y_predicted)
plt.figure(figsize=(10,5))
plt.bar(range(len(first_regression_model.coef_)),first_regression_model.coef_)
plt.show()
print("koeficijenti: ", first_regression_model.coef_)

print("Oblik linearne regresije sa izbacena dva obelezja:\n")
X_train_22 = sm.add_constant(X_train_11.drop('month', axis=1))
X_val_22=sm.add_constant(X_val_11.drop('month', axis=1))
model = sm.OLS(y_train1, X_train_22.astype('float')).fit()
print(model.summary())
third_regression_model = LinearRegression(fit_intercept=True)
third_regression_model.fit(X_train_22, y_train1)
y_predicted = third_regression_model.predict(X_val_22)
model_evaluation(y_val, y_predicted)
plt.figure(figsize=(10,5))
plt.bar(range(len(first_regression_model.coef_)),first_regression_model.coef_)
plt.show()
print("koeficijenti: ", first_regression_model.coef_)

#Medjusobna korelisanost obelezja
numeric_feats = [item for item in x.columns if 'SOF' not in item]
corr_mat = X_train1[numeric_feats].corr()
plt.figure(figsize=(12, 9))
sb.heatmap(corr_mat, annot=True)
plt.show()

#Linearna regresija sa interakcijom izmedju obelezja
print("Linearna regresija sa interakcijom izmedju obelezja:\n")
poly = PolynomialFeatures(interaction_only=True, include_bias=False)
x_inter_train1 = poly.fit_transform(x_train_std)
x_inter_test1 = poly.transform(x_test_std)
print(pd.DataFrame(x_inter_train1).head())
regression_model_inter1 = LinearRegression()
regression_model_inter1.fit(x_inter_train1, y_train1)
y_predicted = regression_model_inter1.predict(x_inter_test1)
model_evaluation(y_val, y_predicted)
plt.figure(figsize=(10,5))
plt.bar(range(len(regression_model_inter1.coef_)),regression_model_inter1.coef_)
plt.show()
print("koeficijenti: ", regression_model_inter1.coef_)

#Linearna regresija sa interakcijom izmedju obelezja-ukljucen kvadrat
print("Linearna regresija sa interakcijom izmedju obelezja-ukljucen kvadrat:\n")
poly = PolynomialFeatures(degree=2,interaction_only=False, include_bias=False)
x_inter_train2 = poly.fit_transform(x_train_std)
x_inter_test2 = poly.transform(x_test_std)
print(pd.DataFrame(x_inter_train2).head())
regression_model_inter2 = LinearRegression()
regression_model_inter2.fit(x_inter_train2, y_train1)
y_predicted = regression_model_inter2.predict(x_inter_test2)
model_evaluation(y_val, y_predicted)
plt.figure(figsize=(10,5))
plt.bar(range(len(regression_model_inter2.coef_)),regression_model_inter2.coef_)
plt.show()
print("koeficijenti: ", regression_model_inter2.coef_)

#Linearna regresija sa interakcijom izmedju obelezja-ukljucen kvadrat i izbacena dva obelezja
print("Linearna regresija sa interakcijom izmedju obelezja-ukljucen kvadrat i izbacena dva obelezja:\n")
poly = PolynomialFeatures(degree=2,interaction_only=False, include_bias=False)
x_inter_train3 = poly.fit_transform(X_train_22)
x_inter_test3 = poly.transform(X_val_22)
print(pd.DataFrame(x_inter_train3).head())
regression_model_inter3 = LinearRegression()
regression_model_inter3.fit(x_inter_train3, y_train1)
y_predicted = regression_model_inter3.predict(x_inter_test3)
model_evaluation(y_val, y_predicted)
plt.figure(figsize=(10,5))
plt.bar(range(len(regression_model_inter3.coef_)),regression_model_inter3.coef_)
plt.show()
print("koeficijenti: ", regression_model_inter3.coef_)

#Ridge regularizacija
# Inicijalizacija
print("Ridge regularizacija:\n")
ridge_model = Ridge(alpha=5)
# Obuka modela
ridge_model.fit(x_inter_train2, y_train1)
# Testiranje
y_predicted = ridge_model.predict(x_inter_test2)
# Evaluacija
model_evaluation(y_val, y_predicted)
# Ilustracija koeficijenata
plt.figure(figsize=(10,5))
plt.bar(range(len(ridge_model.coef_)),ridge_model.coef_)
plt.show()
print("koeficijenti: ", ridge_model.coef_)

#Lasso regularizacija
print("Lasso regularizacija:\n")
lasso_model = Lasso(alpha=0.01)
lasso_model.fit(x_inter_train2, y_train1)
y_predicted = lasso_model.predict(x_inter_test2)
model_evaluation(y_val, y_predicted)

plt.figure(figsize=(10,5))
plt.bar(range(len(lasso_model.coef_)),lasso_model.coef_)
plt.show()
print("koeficijenti: ", lasso_model.coef_)

#Testiranje modela
print("Finalni model:\n")
poly_finall = PolynomialFeatures(degree=2,interaction_only=False, include_bias=False)
x_inter_train2_final = poly.fit_transform(X_train)
x_inter_test2_final = poly.transform(X_test)
print(pd.DataFrame(x_inter_train2).head())
regression_model_inter2_final = LinearRegression()
regression_model_inter2_final.fit(x_inter_train2_final, y_train)
y_predicted = regression_model_inter2_final.predict(x_inter_test2_final)
model_evaluation(y_test, y_predicted)
plt.figure(figsize=(10,5))
plt.bar(range(len(regression_model_inter2_final.coef_)),regression_model_inter2_final.coef_)
plt.show()
print("koeficijenti: ", regression_model_inter2_final.coef_)
lasso_model_final = Lasso(alpha=0.01)
lasso_model_final.fit(x_inter_train2_final, y_train)
y_predicted = lasso_model_final.predict(x_inter_test2_final)
model_evaluation(y_test, y_predicted)

plt.figure(figsize=(10,5))
plt.bar(range(len(lasso_model_final.coef_)),lasso_model_final.coef_)
plt.show()
print("koeficijenti: ", lasso_model_final.coef_)

#KNN klasifikator
#Dodeljivanje labela uzorcima
podaci_KNN=podaci
labele=[]
for vrednost in podaci_KNN['PM_US Post']:
    if vrednost<=55.4: labele.append('bezbedno')
    elif vrednost>=55.5 and vrednost<=150.4: labele.append('nebezbedno')
    else: labele.append('opasno')
podaci_KNN['label'] = labele
imena_klasa=["opasno","nebezbedno","bezbedno"]
podaci_opasno=podaci_KNN[podaci['label']=='opasno']
podaci_nebezbedno=podaci_KNN[podaci['label']=='nebezbedno']
podaci_bezbedno=podaci_KNN[podaci['label']=='bezbedno']
opis_opasno=podaci_opasno.describe()
opis_bezbedno=podaci_bezbedno.describe()
opis_nebezbedno=podaci_nebezbedno.describe()
plt.scatter(podaci_opasno['PM_US Post'],podaci_opasno['TEMP'])
plt.xlabel('Koncentracija čestica')
plt.ylabel('Temperatura')
plt.show()
plt.scatter(podaci_opasno['PM_US Post'],podaci_opasno['DEWP'])
plt.xlabel('Koncentracija čestica')
plt.ylabel('Temperatura rose')
plt.show()
plt.scatter(podaci_opasno['PM_US Post'],podaci_opasno['season'])
plt.xlabel('Koncentracija čestica')
plt.ylabel('Godisnje doba')
plt.show()

#Prikaz boxplot-a za kateogirju bezbedno
boxplot_opasno = podaci_opasno.boxplot(column=['PM_US Post', 'DEWP', 'HUMI','PRES','TEMP','Iws'])  
plt.show()
boxplot_bezbedno = podaci_bezbedno.boxplot(column=['PM_US Post', 'DEWP', 'HUMI','PRES','TEMP','Iws'])  
plt.show()
boxplot_nebezbedno = podaci_nebezbedno.boxplot(column=['PM_US Post', 'DEWP', 'HUMI','PRES','TEMP','Iws'])
plt.show()
sb.distplot(podaci_bezbedno['Iws'],hist=False,fit=norm)


#Podela podataka na trening i test skup
x=podaci_KNN.iloc[:,:-1]
y=podaci_KNN.iloc[:,-1]
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.15,random_state=10,stratify=y)
X_train1,X_val,y_train1,y_val=train_test_split(X_train,y_train,test_size=0.1,random_state=10,stratify=y_train)
#Zastupljenost klasa
print("Zastupljenost klase nebezbedno:",np.sum(y=='nebezbedno'))
print("Zastupljenost klase bezbedno:",np.sum(y=='bezbedno'))
print("Zastupljenost klase opasno:",np.sum(y=='opasno'))

#Odabir parametara
for m in ['euclidean','minkowski','hamming']:
    acc = []
    for i in range (1,13):
        klasifikator=KNeighborsClassifier(n_neighbors=i,metric=m)
        klasifikator.fit(X_train1,y_train1)
        y_pred=klasifikator.predict(X_val)
        c=confusion_matrix(y_val,y_pred)
        print(c)
        acc.append(accuracy_score(y_val, y_pred))
    plt.figure(figsize=(12,6))
    plt.plot(range(1,13),acc,color='blue')
    plt.title("Greska za metriku "+m)
    plt.xlabel("K vrednost")
    plt.ylabel("Tačnost")

#Unakrsna validacija
kf=StratifiedKFold(n_splits=10,shuffle=True, random_state=13)
acc=0
indexes=kf.split(X_train1,y_train1)
fin_conf_mat=np.zeros((len(np.unique(y_train1)),len(np.unique(y_train1))))
for train_index, test_index in indexes:
    X_train=X_train1.iloc[train_index,:]
    X_test=X_train1.iloc[test_index,:]
    y_train=y_train1.iloc[train_index]
    y_test=y_train1.iloc[test_index]
    klasifikator=KNeighborsClassifier(n_neighbors=7,metric='minkowski')
    klasifikator.fit(X_train,y_train)
    y_pred=klasifikator.predict(X_val)
    matrica_konfuzije=confusion_matrix(y_val,y_pred)
    print(matrica_konfuzije)
    acc+=accuracy_score(y_val, y_pred)
    fin_conf_mat+=matrica_konfuzije
print("Finalna matrica je :\n",fin_conf_mat)
print("Prosecna tačnost klasfikatora je:",acc/10)
print("Tačnost klase opasno:",
      (fin_conf_mat[0,0]+fin_conf_mat[1,1]+fin_conf_mat[1,2]+fin_conf_mat[2,1]+fin_conf_mat[2,2])
      /np.sum(fin_conf_mat))
print("Tačnost klase nebezbedno:",
      (fin_conf_mat[1,1]+fin_conf_mat[0,0]+fin_conf_mat[0,2]+fin_conf_mat[2,0]+fin_conf_mat[2,2])
      /np.sum(fin_conf_mat))
print("Tačnost klase bezbedno:",
      (fin_conf_mat[2,2]+fin_conf_mat[0,0]+fin_conf_mat[0,1]+fin_conf_mat[1,0]+fin_conf_mat[1,1])
      /np.sum(fin_conf_mat))

#Obucavanje klasifikatora
klasifikator_konacni=KNeighborsClassifier(n_neighbors=7,metric='minkowski')
klasifikator_konacni.fit(X_train,y_train)
y_pred=klasifikator_konacni.predict(X_test)
konacna_matrica_konfuzije=confusion_matrix(y_test,y_pred)
print(konacna_matrica_konfuzije)
print("Mere uspesnosti klasa")
print("************************************")
print("Tačnost klase opasno:",
      (konacna_matrica_konfuzije[0,0]+konacna_matrica_konfuzije[1,1]+konacna_matrica_konfuzije[1,2]
       +konacna_matrica_konfuzije[2,1]+konacna_matrica_konfuzije[2,2])
      /np.sum(konacna_matrica_konfuzije))
print("Tačnost klase nebezbedno:",
      (konacna_matrica_konfuzije[1,1]+konacna_matrica_konfuzije[0,0]+konacna_matrica_konfuzije[0,2]
       +konacna_matrica_konfuzije[2,0]+konacna_matrica_konfuzije[2,2])
      /np.sum(konacna_matrica_konfuzije))
print("Tačnost klase bezbedno:",
      (konacna_matrica_konfuzije[2,2]+konacna_matrica_konfuzije[0,0]+konacna_matrica_konfuzije[0,1]
       +konacna_matrica_konfuzije[1,0]+konacna_matrica_konfuzije[1,1])
      /np.sum(konacna_matrica_konfuzije))
preciznost_klasa=precision_score(y_test,y_pred,average=None)
print("Preciznost klase opasno:",preciznost_klasa[0])
print("Preciznost klase nebezbedno:",preciznost_klasa[1])
print("Preciznost klase bezbedno:",preciznost_klasa[2])
osetljivost_klasa=recall_score(y_test,y_pred,average=None)
print("Osetljivost klase opasno:",osetljivost_klasa[0])
print("Osetljivost klase nebezbedno:",osetljivost_klasa[1])
print("Osetljivost klase bezbedno:",osetljivost_klasa[2])
f1_klasa=f1_score(y_test,y_pred,average=None)
print("F-score klase opasno:",f1_klasa[0])
print("F-score klase nebezbedno:",f1_klasa[1])
print("F-score klase bezbedno:",f1_klasa[2])
print("************************************")
print("Tačnost klasifikatora:",accuracy_score(y_test, y_pred))
preciznost = precision_score(y_test, y_pred, average = 'macro')
print("Preciznost klasfikatora:",preciznost)
osetljivost = recall_score(y_test, y_pred, average = 'macro')
print("Osetljivost klasifikatora:",osetljivost)
f1 = f1_score(y_test, y_pred, average = 'macro')
print("F-score klasifikatora",f1)


