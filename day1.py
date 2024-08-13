import numpy as np;
import pandas as p;
from sklearn.impute import SimpleImputer;

#Reading Data
dataset=p.read_csv('Data.csv');

#Readind Data
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values

#Handling Misiing Values
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

#Encoding String(Non Binary) Independant Variable
from sklearn.compose import ColumnTransformer;
from sklearn.preprocessing import OneHotEncoder;
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder='passthrough');
X = np.array(ct.fit_transform(X))

#Encoding Dependant Variable
from sklearn.preprocessing import LabelEncoder;
le=LabelEncoder();
Y=le.fit_transform(Y);

#Splitting Dataset into Training Set and Test Set
from sklearn.model_selection import train_test_split;
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=1);


#Make Standardization 
from sklearn.preprocessing import StandardScaler;
sc=StandardScaler();
X_train[:,3:]=sc.fit_transform(X_train[:,3:]);
X_test[:,3:]=sc.fit_transform(X_test[:,3:]);



print(X_train)
print(X_test)



