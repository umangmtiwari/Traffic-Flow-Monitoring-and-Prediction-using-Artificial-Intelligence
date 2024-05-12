import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score, precision_score, confusion_matrix
from sklearn.model_selection import train_test_split
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_excel('traffic-2.xlsx')

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

# Calculate precision
precision = precision_score(y_test, y_pred.round(), average='weighted')
print("Precision:", precision)

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred.round())

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# Plot precision in a graph
plt.figure(figsize=(8, 6))
plt.plot(y_test, label='Actual Total Vehicles')
plt.plot(y_pred, label='Predicted Total Vehicles')
plt.xlabel('Samples')
plt.ylabel('Total Vehicles')
plt.title('Actual vs Predicted Total Vehicles')
plt.legend()
plt.show()
