import numpy as np;
import pandas as pd;
import matplotlib.pyplot as plt;


#Import Files
dataset = pd.read_csv('./data-sets/50_Startups.csv');

#Reading Values
X=dataset.iloc[:,:-1].values;
Y=dataset.iloc[:,-1].values;



#Categorical column
from sklearn.compose import ColumnTransformer;
from sklearn.preprocessing import OneHotEncoder;
ct=ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[3])],remainder='passthrough')
X = np.array(ct.fit_transform(X))


#Splitting Train and Test Data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=0);

#Apply Linear Regresion Model
from sklearn.linear_model import LinearRegression;
model = LinearRegression();
model.fit(X_train,Y_train);



onehot = ct.named_transformers_['encoder']
print("Categories used for encoding:", onehot.categories_)

# Display model coefficients
print("Model coefficients:", model.coef_)


#Predicted Results
Y_pred=model.predict(X_test);

#Setting Precision
np.set_printoptions(precision=2);

#Concatenating the Predicting and Test data for Comparison
print( np.concatenate( ( Y_pred.reshape(len(Y_pred),1) , Y_test.reshape(len(Y_test),1)),1));


predicted=Y_pred;
original=Y_test

indices = np.arange(len(predicted))  # The label locations

# Bar width
width = 0.35

# Create the plot
fig, ax = plt.subplots(figsize=(10, 6))

bars1 = ax.bar(indices - width/2, predicted, width, label='Predicted', color='skyblue')
bars2 = ax.bar(indices + width/2, original, width, label='Original', color='salmon')

# Add labels, title, and custom x-axis tick labels
ax.set_xlabel('Sample Index')
ax.set_ylabel('Values')
ax.set_title('Predicted vs Original Values')
ax.set_xticks(indices)
# ax.set_xticklabels([f'Sample {i+1}' for i in indices])
ax.legend()

# Add a grid and labels to each bar
ax.grid(axis='y', linestyle='--', alpha=0.7) 
# ax.bar_label(bars1, fmt='%.2f', padding=3)
# ax.bar_label(bars2, fmt='%.2f', padding=3)

# Show plot
plt.tight_layout()
plt.savefig('bar_chart_comparison.png')  # Save the plot as a .png file
plt.show()  # Display the plot








