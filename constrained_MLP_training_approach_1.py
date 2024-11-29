import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.losses import MeanAbsolutePercentageError


# Load the dataset
df = pd.read_csv('D:/python_projects/department_power_estimation/Power.csv')

# Preprocess the data
df['Datetime'] = pd.to_datetime(df['Datetime'])
df.set_index('Datetime', inplace=True)

# Extract time features
df['hour'] = df.index.hour
df['day_of_week'] = df.index.dayofweek
df['day_of_month'] = df.index.day
df['month'] = df.index.month

df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# Define input and output columns
input_columns = ['summed_RealPower', 'hour', 'day_of_week', 'day_of_month', 'month', 
                 'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos']
output_columns = df.columns.difference(['summed_RealPower', 'hour', 'day_of_week', 
                                        'day_of_month', 'month', 'hour_sin', 
                                        'hour_cos', 'day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos'])

print(len(output_columns))

# Prepare the input (X) and output (y) arrays
X = df[input_columns].values
y = df[output_columns].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define custom loss function
def custom_loss(y_true, y_pred):
    # Calculate the mean squared error
    mse_loss = MeanSquaredError()(y_true, y_pred)
    mape_loss = MeanAbsolutePercentageError()(y_true, y_pred)
    # Calculate the sum of predictions and the sum of true values
    pred_sum = K.sum(y_pred, axis=1)
    true_sum = K.sum(y_true, axis=1)
    # Mean squared error between the summed predicted and true total power
    sum_loss = K.mean(K.square(true_sum - pred_sum))
    # Total loss is a combination of both terms
    return mape_loss + sum_loss

# Define the model with softmax and Lambda layer for scaled proportional outputs
model = Sequential([
    Dense(1000, activation='relu', input_shape=(len(input_columns),)),
    Dense(1000, activation='relu'),
    Dense(1000, activation='relu'),
    Dense(1000, activation='relu'),
    Dense(1000, activation='relu'),
    Dense(len(output_columns), activation='linear')  # One output per building
])

# Compile the model with the custom loss function
learning_rate = 0.001
optimizer = Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss=custom_loss, metrics=['mae'])

# Train the model
history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split=0.2)

# Predict on the test set
y_pred = model.predict(X_test)

# Calculate the MAE and MSE for each building
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)

# Calculate the MAE and MSE between the predicted total and actual summed_RealPower
predicted_totals = y_pred.sum(axis=1)
actual_totals = X_test[:, 0]
total_mae = mean_absolute_error(actual_totals, predicted_totals)
total_mse = mean_squared_error(actual_totals, predicted_totals)
print("Total Power MAE:", total_mae)
print("Total Power MSE:", total_mse)

# Define a function for MAPE
def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true), axis=0) * 100

# Compute MAPE per building
mape_per_building = calculate_mape(y_test, y_pred)
average_mape = np.mean(mape_per_building)
print('MAPE per building:', mape_per_building)
print('Average MAPE:', average_mape)

# Plot actual vs. predicted power consumption for each building
building_names = output_columns
r2_list = []
mse_list = []
mape_list = []
mae_list = []
building_dict = dict()
for i, building in enumerate(building_names):
    building_dict[building] = dict()
    r2 = r2_score(y_test[:, i], y_pred[:, i])
    r2_list.append(r2)
    mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
    mae_list.append(mae)
    mape = calculate_mape(y_test[:, i], y_pred[:, i])
    mape_list.append(mape)
    mse = mean_squared_error(y_test[:, i], y_pred[:, i])
    mse_list.append(mse)
    plt.figure(figsize=(10, 6))
    plt.plot(y_test[:, i], label="Actual Power", color="blue")
    plt.plot(y_pred[:, i], label="Predicted Power", color="orange", alpha=0.7)
    plt.title(f"Power Consumption Prediction for {building} (R² = {r2:.3f})")
    plt.xlabel("Time Index")
    plt.ylabel("Power Consumption")
    plt.legend()
    plt.savefig(f"./department_power_estimation/plots_MLP/{building}_power_prediction_r2_{r2:.3f}.png")
    plt.close()
    print(f"R² score for {building}: {r2:.3f}")

# Print average evaluation metrics
print()
print('Average R²:', np.mean([r2_score(y_test[:, i], y_pred[:, i]) for i in range(len(building_names))]))
print('Average MAE:', np.mean([mean_absolute_error(y_test[:, i], y_pred[:, i]) for i in range(len(building_names))]))
print('Average MSE:', np.mean([mean_squared_error(y_test[:, i], y_pred[:, i]) for i in range(len(building_names))]))
print('Average MAPE:', average_mape)
