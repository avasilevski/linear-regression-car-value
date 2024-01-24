from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('./car_resale_value.csv')

# Prepare the data
X = data[['Mileage (thousands of miles)']]  # Features matrix must be 2D for scikit-learn
y = data['Resale Value (thousands of dollars)']  # Target variable

# Initialize the model
model = LinearRegression()

# Fit the model
model.fit(X, y)

# Print the coefficients
print("Intercept:", model.intercept_)
print("Slope:", model.coef_[0])

# Predict values
y_pred = model.predict(X)

# Plot the results
plt.scatter(X, y, color='blue', label='Training Data')
plt.plot(X, y_pred, color='red', label='Linear Regression')
plt.xlabel('Mileage (thousands of miles)')
plt.ylabel('Resale Value (thousands of dollars)')
plt.title('Linear Regression Fit with scikit-learn')
plt.legend()
plt.show()
