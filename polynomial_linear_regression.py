
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('./data-sets/Position_Salaries.csv');

X=dataset.iloc[:,1:-1].values;
y=dataset.iloc[:,-1].values;

#Checking the data graph
# plt.scatter(X,y,color='red');
# plt.show()

#traning model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y);

#making polynomial features
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=4);
X_poly=poly_reg.fit_transform(X);

#training model on polynomial feautres
poly_model=LinearRegression();
poly_model.fit(X_poly,y);


#Model with simple linear and polynomial linear regression
plt.scatter(X,y,color='red');
plt.plot(X,model.predict(X),color='blue')
plt.plot(X,poly_model.predict(X_poly),color='green');
plt.xlabel('Job Level')
plt.ylabel('Salary')
plt.show();


#creating data-set with 0.1 step for smoother curve
plt.scatter(X,y,color='red');
X_grid=np.arange(min(X),max(X),0.1);
X_grid=X_grid.reshape((len(X_grid),1));
X_data=poly_reg.fit_transform(X_grid)
plt.plot(X_grid,poly_model.predict(X_data),color='black');
plt.show()

#predictions with linear model
print(model.predict([[6.5]]))
#predictions with polynomail linear regression
print(poly_model.predict(poly_reg.fit_transform([[6.5]])))
