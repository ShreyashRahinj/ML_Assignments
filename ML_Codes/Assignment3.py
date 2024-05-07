import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error

dataset = pd.read_csv('manhattan.csv')
x = dataset.iloc[:,2:-2].values
y = dataset.iloc[:,1].values

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=0)

regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

# Find Accuracy Score
print(f"R-Squared Score : {r2_score(y_test,y_pred)}")
print(f"MSE : {mean_squared_error(y_test,y_pred)}")

# Plot the linear model
plt.scatter(y_test, y_pred, color='red')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted')
plt.show()