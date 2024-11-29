import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np


def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true), axis=0) * 100

df = pd.read_csv('D:/python_projects/department_power_estimation/Power.csv')
print(df.head())
# Extract datetime features

df['Datetime'] = pd.to_datetime(df['Datetime'])
df['hour'] = df['Datetime'].dt.hour
df['dayofweek'] = df['Datetime'].dt.dayofweek
df['month'] = df['Datetime'].dt.month
df['dayofmonth'] = df['Datetime'].dt.day

# X: Input features - total power + datetime-related features
X = df[['summed_RealPower', 'hour', 'dayofweek', 'month', 'dayofmonth']]

# y: Target is the power consumption of each building (drop 'Datetime' and 'summed_RealPower')
y = df.drop(columns=['Datetime', 'summed_RealPower', 'hour', 'dayofweek', 'month', 'dayofmonth'])

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Train an XGBoost model for each building
r2_scores = {}

r2_list = []
mse_list = []
mape_list = []
mae_list = []
building_dict = dict()
for i, column in enumerate(y.columns):
    building_dict[column] = dict()
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, learning_rate=0.01, max_depth=6)
    model.fit(X_train, y_train[column])
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate R² score
    r2 = r2_score(y_test[column], y_pred)
    r2_scores[column] = r2
    r2_list.append(r2)
    mae = mean_absolute_error(np.array(y_test[column]), y_pred)
    mae_list.append(mae)
    mape = calculate_mape(np.array(y_test[column]), y_pred)
    mape_list.append(mae)
    mse = mean_squared_error(np.array(y_test[column]), y_pred)
    mse_list.append(mse)

    building_dict[column]['r2'] = r2
    building_dict[column]['mae'] = mae
    building_dict[column]['mape'] = mape
    building_dict[column]['mse'] = mse
    
    # Plot actual vs predicted values for each building
    plt.figure(figsize=(10, 6))
    plt.plot(y_test.index, y_test[column], label='Actual')
    plt.plot(y_test.index, y_pred, label='Predicted')
    plt.title(f"Power Consumption Prediction for {column}")
    plt.xlabel('Time')
    plt.ylabel('Power Consumption')
    plt.legend()
    plt.savefig(f"./department_power_estimation/XGBoost_plots/{column}_power_prediction_r2_{r2:.3f}.png")

# Print R² scores for each building
print("R² Scores for each building:")
for building, r2 in r2_scores.items():
    print(f"{building}: {r2:.4f}")

print()
