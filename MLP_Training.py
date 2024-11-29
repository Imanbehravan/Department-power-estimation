import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


df = pd.read_csv('D:/python_projects/department_power_estimation/Power.csv')
print(df.head())

# Convert the 'Datetime' column to a datetime format
df['Datetime'] = pd.to_datetime(df['Datetime'])

# Set Datetime as the index (optional, but can be useful for time series analysis)
df.set_index('Datetime', inplace=True)


# Extract useful time features
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


# Separate the columns that need scaling (all except the datetime-based features)
# columns_to_scale = df.columns.difference(['hour', 'day_of_week', 'day_of_month', 'month'])
# scaler = MinMaxScaler()

# # Scale the power consumption values
# df[columns_to_scale] = scaler.fit_transform(df[columns_to_scale])

# (Optional) Save the scaler if you plan to use it to inverse-transform predictions later

input_columns = ['summed_RealPower', 'hour', 'day_of_week', 'day_of_month', 'month', 'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos']
# Define output columns (all buildings)
output_columns = df.columns.difference(['Datetime', 'summed_RealPower', 'hour', 'day_of_week', 'day_of_month', 'month', 'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos'])

# Extract input (X) and output (y) arrays
X = df[input_columns].values
y = df[output_columns].values

# Split into training and testing sets (example split)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = Sequential([
    Dense(1000, activation='relu', input_shape=(len(input_columns),)),
    Dense(1000, activation='relu'),
    Dense(1000, activation='relu'),
    Dense(1000, activation='relu'),
    Dense(1000, activation='relu'),
    Dense(len(output_columns), activation='linear')  # One output per building
])

# model = Sequential([
#     Dense(5, activation='relu', input_shape=(len(input_columns),)),
#     Dense(len(output_columns), activation='linear')  # One output per building
# ])

learning_rate = 0.001  # Adjust this value as needed
optimizer = Adam(learning_rate=learning_rate)

# Compile the model
model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])

# Train the model
history = model.fit(X_train, y_train, epochs=200, batch_size=32, validation_split=0.2)

y_pred  = model.predict(X_test)


# Calculate the MAE and MSE for each building
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print()


# Check if the sum of predicted values for each row is close to the input total power
predicted_totals = y_pred.sum(axis=1)
actual_totals = X_test[:, 0]  # Assuming `summed_RealPower` is the first column in X_test

# Calculate the MAE and MSE between predicted totals and actual totals
total_mae = mean_absolute_error(actual_totals, predicted_totals)
total_mse = mean_squared_error(actual_totals, predicted_totals)

print("Total Power MAE:", total_mae)
print("Total Power MSE:", total_mse)


def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true), axis=0) * 100

# Compute MAPE per building
mape_per_building = calculate_mape(y_test, y_pred)

average_mape = np.mean(mape_per_building)

print('mape_per_building', mape_per_building)
print('average_mape', average_mape)

# Plot actual vs. predicted power consumption for a single building, e.g., the first building
building_names = output_columns  # or provide a list of building names if they are not in the column headers
r2_list = []
mse_list = []
mape_list = []
mae_list = []
building_dict = dict()
# Loop through each building and create a separate plot
for i, building in enumerate(building_names):
    building_dict[building] = dict()
    # Calculate the R² score for the current building
    r2 = r2_score(y_test[:, i], y_pred[:, i])
    r2_list.append(r2)
    mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
    mae_list.append(mae)
    mape = calculate_mape(y_test[:, i], y_pred[:, i])
    mape_list.append(mae)
    mse = mean_squared_error(y_test[:, i], y_pred[:, i])
    mse_list.append(mse)

    building_dict[building]['r2'] = r2
    building_dict[building]['mae'] = mae
    building_dict[building]['mape'] = mape
    building_dict[building]['mse'] = mse

    # Create a plot for the current building
    plt.figure(figsize=(10, 6))
    plt.plot(y_test[:, i], label="Actual Power", color="blue")
    plt.plot(y_pred[:, i], label="Predicted Power", color="orange", alpha=0.7)
    
    # Include the R² score in the plot title
    plt.title(f"Power Consumption Prediction for {building} (R² = {r2:.3f})")
    plt.xlabel("Time Index")
    plt.ylabel("Power Consumption")
    plt.legend()

    # Save the plot as an image file
    plt.savefig(f"./department_power_estimation/plots_MLP/{building}_power_prediction_r2_{r2:.3f}.png")
    plt.close()  # Close the figure after saving to avoid displaying it in notebooks

    # Print the R² score for each building
    print(f"R² score for {building}: {r2:.3f}")

print()
print('average r2: ', np.mean(r2_list))
print('average mae: ', np.mean(mae_list))
print('average mse: ', np.mean(mse_list))
print('average mape: ', np.mean(mape_list))
print()
