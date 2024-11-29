import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('D:/python_projects/department_power_estimation/Power.csv')

# Convert 'Datetime' to datetime format and set as index
df['Datetime'] = pd.to_datetime(df['Datetime'])
df.set_index('Datetime', inplace=True)

# Extract time features
df['hour'] = df.index.hour
df['day_of_week'] = df.index.dayofweek
df['day_of_month'] = df.index.day
df['month'] = df.index.month

# Cyclical encoding for time-based features
df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# Define input and output features
input_columns = ['summed_RealPower', 'hour', 'day_of_week', 'day_of_month', 'month', 'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos']
output_columns = df.columns.difference(['summed_RealPower', 'hour', 'day_of_week', 'day_of_month', 'month', 'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos', 'month_sin', 'month_cos'])

# Prepare input (X) and output (y) arrays
X = df[input_columns].values
y = df[output_columns].values

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define and train an SVR model for each building
svr_models = []
y_pred = np.zeros(y_test.shape)

for i, building in enumerate(output_columns):
    # Initialize SVR for each building
    svr = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    svr.fit(X_train, y_train[:, i])
    svr_models.append(svr)

    # Predict for the test set
    y_pred[:, i] = svr.predict(X_test)

# Calculate MAE and MSE for each building
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

print("Mean Absolute Error (MAE):", mae)
print("Mean Squared Error (MSE):", mse)
print()

# Calculate MAE and MSE for total power
predicted_totals = y_pred.sum(axis=1)
actual_totals = X_test[:, 0]  # Assuming `summed_RealPower` is the first column in X_test

total_mae = mean_absolute_error(actual_totals, predicted_totals)
total_mse = mean_squared_error(actual_totals, predicted_totals)

print("Total Power MAE:", total_mae)
print("Total Power MSE:", total_mse)

# Calculate MAPE per building
def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true), axis=0) * 100

mape_per_building = calculate_mape(y_test, y_pred)
average_mape = np.mean(mape_per_building)

print('MAPE per building:', mape_per_building)
print('Average MAPE:', average_mape)

r2_list = []
mse_list = []
mape_list = []
mae_list = []
building_dict = dict()
# Plot actual vs. predicted power consumption for each building
for i, building in enumerate(output_columns):
    building_dict[building] = dict()
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

    
    plt.figure(figsize=(10, 6))
    plt.plot(y_test[:, i], label="Actual Power", color="blue")
    plt.plot(y_pred[:, i], label="Predicted Power", color="orange", alpha=0.7)
    
    plt.title(f"Power Consumption Prediction for {building} (R² = {r2:.3f})")
    plt.xlabel("Time Index")
    plt.ylabel("Power Consumption")
    plt.legend()
    plt.savefig(f"./department_power_estimation/plots_SVR/{building}_power_prediction_r2_{r2:.3f}.png")
    plt.close()

    print(f"R² score for {building}: {r2:.3f}")


print()
print('average r2: ', np.mean(r2_list))
print('average mae: ', np.mean(mae_list))
print('average mse: ', np.mean(mse_list))
print('average mape: ', np.mean(mape_list))
print()
