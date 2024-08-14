import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;

#Importing the CSV File
dataset=pd.read_csv('./data-sets/Salary_Data.csv');

#Reading the data(X means Year Experience (first row) and Y means Salary (last row))
X=dataset.iloc[:,:-1].values;
Y=dataset.iloc[:,-1].values;

#Splitting Train and Test Data-sets
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=1);

#Training the model with simple Linear Regression
from sklearn.linear_model import LinearRegression;
regressor=LinearRegression();
regressor.fit(X_train,Y_train);

#Predicting the results by giving the Sample Test Data;
Y_predict=regressor.predict(X_test)

#Visualizing the Train Data
plt.scatter(X_train,Y_train,color='red');
plt.plot(X_train,regressor.predict(X_train),color='blue');
plt.title('Salary vs Experience');
plt.xlabel('Years of Experience');
plt.ylabel('Salary');
plt.show();

#Visualizing the Test Data
plt.scatter(X_test,Y_test,color='red');
plt.plot(X_train,regressor.predict(X_train),color='blue');
plt.title('Salary vs Experience');
plt.xlabel('Years of Experience');
plt.ylabel('Salary');
plt.show();





