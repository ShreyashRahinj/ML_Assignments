import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def max_absolute_scaling(data):
    maxi = np.max(data)
    return data / maxi


def min_max(data):
    mini = min(data)
    maxi = max(data)

    return [(v - mini) / (maxi - mini) for v in data]


def normalization(data):
    mean = np.mean(data)
    std = np.std(data)

    return [(v - mean) / std for v in data]


df = pd.read_csv('starbucks_customers.csv')
avg_spent = df['avg_spent']

max_scaled = max_absolute_scaling(avg_spent)
min_max_scaled = min_max(avg_spent)
normalized = normalization(avg_spent)

plt.plot(avg_spent,max_scaled)
plt.show()

plt.plot(avg_spent,min_max_scaled)
plt.show()

plt.plot(avg_spent,normalized)
plt.show()
