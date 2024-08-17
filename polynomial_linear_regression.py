
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset=pd.read_csv('./data-sets/Position_Salaries.csv');

X=dataset.iloc[:,1:-1].values;
y=dataset.iloc[:,-1].values;

from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X, y)

