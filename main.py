# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Load the diabetes dataset
diabetes = load_diabetes()

# Convert it to a pandas DataFrame
data = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
data['progression'] = diabetes.target

# Plot BMI vs Progression
plt.figure(figsize=(8, 6))
plt.scatter(data['bmi'], data['progression'], alpha=0.6)
plt.xlabel('BMI')
plt.ylabel('Disease Progression')
plt.title('Scatter Plot of BMI vs Disease Progression')
plt.show()

# Prepare data for the model
X = data.drop('progression', axis=1)  # Features
y = data['progression']  # Target

# Split the data into training and validation sets
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

# Initialize and fit the Random Forest Regressor model
diabetes_model = RandomForestRegressor(random_state=1)
diabetes_model.fit(train_X, train_y)

# Make predictions on a small sample
print("The predictions on the first few samples are:")
sample_predictions = diabetes_model.predict(X.head())
print(sample_predictions)

# Predict on the validation set and evaluate the model
val_predictions = diabetes_model.predict(val_X)
mae = mean_absolute_error(val_y, val_predictions)
print(f"Mean Absolute Error on the validation set: {mae:.2f}")
