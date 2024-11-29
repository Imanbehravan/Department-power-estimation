import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Multiply
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.losses import MeanAbsolutePercentageError


def calculate_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true), axis=0) * 100
# Load and preprocess the data
df = pd.read_csv('/home/iman/projects/kara/Projects/department_power_estimation/Power.csv')
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
output_columns = df.columns.difference(input_columns)

# Separate inputs and outputs
X = df[input_columns].values
y = df[output_columns].values

# Normalize input features (excluding summed_RealPower)
scaler = StandardScaler()
X[:, 1:] = scaler.fit_transform(X[:, 1:])

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the model
time_features_input = Input(shape=(X_train.shape[1] - 1,), name="time_features_input")  # Exclude summed_RealPower
summed_real_power_input = Input(shape=(1,), name="summed_real_power_input")  # Summed power as a separate input

hidden = Dense(1000, activation='relu')(time_features_input)
hidden = Dense(1000, activation='relu')(hidden)
hidden = Dense(1000, activation='relu')(hidden)
hidden = Dense(1000, activation='relu')(hidden)
output_proportions = Dense(y_train.shape[1], activation='softmax', name="proportions_output")(hidden)  # Softmax layer

# Scale the output proportions by summed_RealPower
output_scaled = Multiply(name="scaled_output")([output_proportions, summed_real_power_input])

# Combine into a model
model = Model(inputs=[time_features_input, summed_real_power_input], outputs=output_scaled)

# Compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
history = model.fit(
    [X_train[:, 1:], X_train[:, 0]],  # Exclude summed_RealPower from time features
    y_train,
    validation_split=0.2,
    epochs=200,
    batch_size=32,
    verbose=1
)

# Evaluate the model
test_loss, test_mae = model.evaluate([X_test[:, 1:], X_test[:, 0]], y_test, verbose=0)
print(f"Test Loss: {test_loss}, Test MAE: {test_mae}")



# Predict on test data
y_pred = model.predict([X_test[:, 1:], X_test[:, 0]])
mape_per_building = calculate_mape(y_test, y_pred)
average_mape = np.mean(mape_per_building)
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
    plt.savefig(f"/home/iman/projects/kara/Projects/department_power_estimation/MLP_plots/{building}_power_prediction_r2_{r2:.3f}.png")
    plt.close()
    print(f"R² score for {building}: {r2:.3f}")

# Print average evaluation metrics
print()
print('Average R²:', np.mean([r2_score(y_test[:, i], y_pred[:, i]) for i in range(len(building_names))]))
print('Average MAE:', np.mean([mean_absolute_error(y_test[:, i], y_pred[:, i]) for i in range(len(building_names))]))
print('Average MSE:', np.mean([mean_squared_error(y_test[:, i], y_pred[:, i]) for i in range(len(building_names))]))
print('Average MAPE:', average_mape)