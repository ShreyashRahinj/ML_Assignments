import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,f1_score,confusion_matrix,precision_score,recall_score

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt"
names = ["Variance","Skewness","Curtosis","Entropy","Class"]
data = pd.read_csv(url, names=names)

X = data.drop("Class",axis=1)
y = data["Class"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=69)

classifier = RandomForestClassifier(random_state=69)
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)
print(f"Accuracy Score : {accuracy_score(y_test,y_pred)}")
print(f"F1 Score : {f1_score(y_test,y_pred)}")
print(f"Confusion Matrix : {confusion_matrix(y_test,y_pred)}")
print(f"Precision Score : {precision_score(y_test,y_pred)}")
print(f"Recall Score : {recall_score(y_test,y_pred)}")