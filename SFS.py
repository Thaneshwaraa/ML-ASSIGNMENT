# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Step 2: Create a synthetic dataset
np.random.seed(42)
data = {
    'engine_size': np.random.randint(100, 1000, 1000),
    'mileage': np.random.randint(1000, 50000, 1000),
    'brand': np.random.choice(['Yamaha', 'Honda', 'Suzuki', 'Ducati', 'BMW'], 1000),
    'year': np.random.randint(2000, 2022, 1000),
    'price': np.random.randint(1000, 15000, 1000)
}

df = pd.DataFrame(data)

# Convert categorical brand into dummy variables
df = pd.get_dummies(df, columns=['brand'], drop_first=True)

# Separate features and target
X = df.drop('price', axis=1)
y = df['price']

# Step 3: Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Initialize a linear regression model
model = LinearRegression()

# Step 5: Perform Sequential Feature Selection
sfs = SequentialFeatureSelector(model, n_features_to_select=3, direction='forward')
sfs.fit(X_train, y_train)

# Get the selected features
selected_features = X.columns[sfs.get_support()]
print("Selected features:", selected_features)

# Step 6: Train the model with the selected features
model.fit(X_train[selected_features], y_train)

# Predict on the test set
y_pred = model.predict(X_test[selected_features])

# Calculate and print Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error with selected features: {mse:.2f}")

# Step 7: Plotting actual vs predicted prices
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Bike Prices')
plt.show()

# Step 8: Plotting a pie chart of feature importance
coefficients = model.coef_
plt.figure(figsize=(8, 8))
plt.pie(np.abs(coefficients), labels=selected_features, autopct='%1.1f%%', startangle=140)
plt.title('Feature Importance based on Model Coefficients')
plt.show()
