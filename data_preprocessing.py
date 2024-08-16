import numpy as np;
import pandas as p;
from sklearn.impute import SimpleImputer;
import matplotlib.pyplot as plt

#Importing Data-set File
dataset=p.read_csv('./data-sets/Data.csv');

#Readind Data(: means all rows and :-1 means every column except last one)
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values

#Handling Misiing Values (git means prepare data and transform meand execute it)
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

#Encoding String(Non Binary) Independant Variable
# Encoding categorical data(transforimg country names into some numerical values )
from sklearn.compose import ColumnTransformer;
from sklearn.preprocessing import OneHotEncoder;
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder='passthrough');
X = np.array(ct.fit_transform(X))

#Encoding Dependant Variable
#Trasnforming Yes,No into comparable 0,1
from sklearn.preprocessing import LabelEncoder;
le=LabelEncoder();
Y=le.fit_transform(Y);

#Splitting Dataset into Training Set and Test Set(Splitting the data-set)
from sklearn.model_selection import train_test_split;
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=1);


#Make Standardization 
from sklearn.preprocessing import StandardScaler;
sc=StandardScaler();
X_train[:,3:]=sc.fit_transform(X_train[:,3:]);
X_test[:,3:]=sc.fit_transform(X_test[:,3:]);
  












