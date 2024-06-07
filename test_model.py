import joblib
import numpy as np
model = joblib.load('model.joblib')
X_new = np.array([[3, 5]])
predictions = model.predict(X_new)
print(f"Predictions: {predictions}")
