# The Dataset is a list of brain weight and body weigth from a bunch of animals in a form of csv file. Create linear regression model and perform the following:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error

dataset = pd.read_csv('Brain_Body.csv')

# Define Independent and Dependent variables 
brain_weight = dataset.iloc[:,:-1].values # Independent
body_weight = dataset.iloc[:,-1].values # Dependent

# Split Dataset
x_train,x_test,y_train,y_test = train_test_split(brain_weight,body_weight,random_state=0)

regressor = LinearRegression()
regressor.fit(x_train,y_train)

# Perform the prediction
y_pred = regressor.predict(x_test)

# Find Accuracy Score
print(r2_score(y_test,y_pred))
print(mean_squared_error(y_test,y_pred))

# Plot the linear model
plt.scatter(x_test,y_pred,color = 'red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title('Brain_Weight vs Body_Weight')
plt.xlabel('Brain')
plt.ylabel('Body')
plt.show()