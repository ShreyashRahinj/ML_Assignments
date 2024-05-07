# Importing the Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Import the Dataset
dataset = pd.read_csv('Heights_Weights.csv')
heights = dataset.iloc[:,:-1].values
weights = dataset.iloc[:,-1].values

# Training the Model on Dataset
regressor = LinearRegression()
regressor.fit(heights,weights)

# Predict Weight using model
weights_pred = regressor.predict(heights)

# Visualising the clusters
plt.scatter(heights,weights,color = 'red')
plt.plot(heights,weights_pred,color='blue')
plt.title('Heights vs Weights')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.show()