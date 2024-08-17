import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Import Files
dataset = pd.read_csv('./data-sets/dataset_multicollinearity_effect.csv')

# Reading Values
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

# Without dropping a dummy variable
ct_no_drop = ColumnTransformer(transformers=[('encoder', OneHotEncoder(drop=None), [3])], remainder='passthrough')
X_no_drop = np.array(ct_no_drop.fit_transform(X))

# With dropping the first dummy variable
ct_drop = ColumnTransformer(transformers=[('encoder', OneHotEncoder(drop='first'), [3])], remainder='passthrough')
X_drop = np.array(ct_drop.fit_transform(X))

# Splitting Train and Test Data for both versions
X_train_no_drop, X_test_no_drop, Y_train, Y_test = train_test_split(X_no_drop, Y, test_size=0.2, random_state=0)
X_train_drop, X_test_drop, Y_train, Y_test = train_test_split(X_drop, Y, test_size=0.2, random_state=0)

# Fit models
model_no_drop = LinearRegression()
model_drop = LinearRegression()

model_no_drop.fit(X_train_no_drop, Y_train)
model_drop.fit(X_train_drop, Y_train)

# Predictions
Y_pred_no_drop = model_no_drop.predict(X_test_no_drop)
Y_pred_drop = model_drop.predict(X_test_drop)

print(Y_pred_drop)
print(Y_pred_no_drop)

# Compare the predictions
are_predictions_close = np.allclose(Y_pred_no_drop, Y_pred_drop)  # This checks if the predictions are approximately equal

print("Are predictions approximately equal?", are_predictions_close)
