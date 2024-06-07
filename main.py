import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
y = np.dot(X, np.array([1, 2])) + 3

model = LinearRegression().fit(X, y)

joblib.dump(model, 'model.joblib')

print("Model trained and saved as model.joblib")
