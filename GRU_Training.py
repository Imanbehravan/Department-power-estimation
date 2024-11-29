import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
import matplotlib.pyplot as plt

# Step 1: Data Loading and Preprocessing
# Assuming your dataset is loaded in a DataFrame df
# df = pd.read_csv("your_dataset.csv")  # Load your dataset


df = pd.read_csv('D:/python_projects/department_power_estimation/Power.csv')
print(df.head())

# Extract datetime features
df['Datetime'] = pd.to_datetime(df['Datetime'])
df['hour'] = df['Datetime'].dt.hour
df['dayofweek'] = df['Datetime'].dt.dayofweek
df['month'] = df['Datetime'].dt.month
df['dayofmonth'] = df['Datetime'].dt.day

# Drop 'Datetime' column and 'summed_RealPower' from features
X = df[['summed_RealPower', 'hour', 'dayofweek', 'month', 'dayofmonth']]
# The target is the power consumption of each building (e.g., CenterHall, EastCampus, etc.)
y = df.drop(columns=['Datetime', 'summed_RealPower', 'hour', 'dayofweek', 'month', 'dayofmonth'])

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Reshape data to fit the GRU input shape (samples, time steps, features)
X_train = X_train.values.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = X_test.values.reshape(X_test.shape[0], 1, X_test.shape[1])

# Step 3: Build the GRU model
model = Sequential()

# GRU layer with 50 units
model.add(GRU(units=50, input_shape=(X_train.shape[1], X_train.shape[2])))

# Output layer (13 outputs for each building)
model.add(Dense(units=y_train.shape[1]))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Step 4: Train the GRU model
history = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_test, y_test), verbose=1)

# Step 5: Evaluate the model
y_pred = model.predict(X_test)

# Step 6: Calculate R² score for each building (target)
r2_scores = {}
for i, column in enumerate(y.columns):
    r2_scores[column] = r2_score(y_test.iloc[:, i], y_pred[:, i])

print("R² Scores for each building:")
for building, r2 in r2_scores.items():
    print(f"{building}: {r2:.4f}")

# Step 7: Plot predictions vs actuals for each building
for i, column in enumerate(y.columns):
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.index, y_test.iloc[:, i], label='Actual')
    plt.plot(y_test.index, y_pred[:, i], label='Predicted')
    plt.title(f"Power Consumption Prediction for {column}")
    plt.xlabel('Time')
    plt.ylabel('Power Consumption')
    plt.legend()
    plt.savefig(f"./department_power_estimation/GRU_plots/{column}_power_prediction_r2_{r2:.3f}.png")


