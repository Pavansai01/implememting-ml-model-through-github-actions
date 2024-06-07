# model.py
import numpy as np
from sklearn.linear_model import LinearRegression
import joblib

# Create sample data
# X is the input features matrix
X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
# y is the target vector
y = np.dot(X, np.array([1, 2])) + 3

# Train a linear regression model
model = LinearRegression().fit(X, y)

# Print the coefficients and intercept of the trained model
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")

# Save the model to a file
joblib.dump(model, 'model.joblib')

print("Model trained and saved as model.joblib")
