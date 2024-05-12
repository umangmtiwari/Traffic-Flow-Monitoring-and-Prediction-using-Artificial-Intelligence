import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from datetime import datetime

# Load the dataset
df = pd.read_excel('traffic-4.xlsx')

# Convert 'Time' column to datetime format
df['Time'] = pd.to_datetime(df['Time'])

# Extract features and target variable
X = df[['Car Count', 'Bike Count', 'Truck Count', 'Bus Count']]
y = df['Total Vehicles']

# Extract hour and minute from 'Time' as numerical features
df['Hour'] = df['Time'].dt.hour
df['Minute'] = df['Time'].dt.minute

# Drop 'Time' from the features
X = df[['Car Count', 'Bike Count', 'Truck Count', 'Bus Count', 'Hour', 'Minute']]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocess categorical variables and scale numerical variables
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), ['Car Count', 'Bike Count', 'Truck Count', 'Bus Count', 'Hour', 'Minute']),
    ('cat', OneHotEncoder(), [])  # No categorical variables in this case
])

# Define the model
model = make_pipeline(preprocessor, LinearRegression())

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Calculate R^2 (coefficient of determination)
r2 = r2_score(y_test, y_pred)
print("R^2 (Coefficient of Determination):", r2)

# Example of making a prediction for a new data point
new_data = pd.DataFrame({
    'Car Count': [15],
    'Bike Count': [3],
    'Truck Count': [3],
    'Bus Count': [16],
    'Hour': [11],
    'Minute': [50]
})

# Make prediction for new data
prediction = model.predict(new_data)

# Round the predicted total vehicles
predicted_total_vehicles = round(prediction[0])
print("Predicted Total Vehicles:", predicted_total_vehicles)

# Extract hour and minute for the predicted time
predicted_hour = int(new_data['Hour'].iloc[0])
predicted_minute = int(new_data['Minute'].iloc[0])

print("Predicted Time of Traffic:", f"{predicted_hour:02d}:{predicted_minute:02d}")

# Get the category of vehicle causing traffic
vehicle_categories = ['Car', 'Bike', 'Truck', 'Bus']
category_index = model.named_steps['linearregression'].coef_[:-2].argmax()
predicted_category = vehicle_categories[category_index]
print("Predicted Category of Vehicle Causing Traffic:", predicted_category)
