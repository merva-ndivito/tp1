import pandas as pd
import sklearn
from  sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib 

#importation du ficher csv
dataset=pd.read_csv('datadata.csv')

#determinier les x et y
X=dataset[['longuer','couleur','qualite']]
y=dataset['prix']

#on subdiviser le dataset en 2 partie

Xtrain,Xtest,yTrain,yTest=train_test_split(X,y,test_size=0.2,random_state=0)

#choisir l'algorythme
regresseur=LinearRegression()

#entrainer grace a la methode fit
regresseur.fit(Xtrain,yTrain)

#enregistre le modele
joblib.dump(regresseur, 'model.pkl')

#predire ou faire le test
yPred=regresseur.predict(Xtest)

df=pd.DataFrame({
    'prix estime' :yTest,
    'prix predit':yPred
})
print(df)