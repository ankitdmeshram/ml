import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the dataset from a CSV file
data = pd.read_csv('ass3.csv')
print(data)


# Assuming your dataset has two columns: 'X' for independent variable and 'y' for dependent variable
X = data['x'].values
print("Independet Variable: ",X)

y = data['y'].values
print("Dependent Variable: ",y)



# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit a Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

print("Train dataset : ",X_train)

# Get the coefficients (slope and intercept) of the linear regression model
slope = model.coef_[0][0]
intercept = model.intercept_[0]

print(f"Slope: {slope}")
print(f"Intercept: {intercept}")

# Make predictions on the test data
y_pred = model.predict(X_test)

'''#Plot the training data
plt.figure(figsize=(8, 6))
plt.scatter(X_train, y_train, color='blue', label='Training Data')
plt.plot(X_train, model.predict(X_train), color='red', linewidth=2, label='Linear Regression')
plt.title('Training Data and Linear Regression Model')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()

# Plot the test data
plt.figure(figsize=(8, 6))
plt.scatter(X_test, y_test, color='blue', label='Test Data')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Predictions')
plt.title('Test Data and Predictions')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
'''
#Take new input for prediction
new_value = float(input("Enter a new value for prediction: "))
predicted_y = model.predict(np.array([[new_value]]))

print(f"Predicted y for X={new_value}: {predicted_y[0][0]}")
