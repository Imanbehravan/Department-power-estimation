import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

# Assuming the data is loaded into a DataFrame `df`
# Assuming the dataset has a 'summed_RealPower' column and other columns for building power consumption

df = pd.read_csv('D:/python_projects/department_power_estimation/Power.csv')
print(df.head())

# Extract the relevant columns
df['Datetime'] = pd.to_datetime(df['Datetime'])
df['hour'] = df['Datetime'].dt.hour
df['day_of_week'] = df['Datetime'].dt.dayofweek
df['month'] = df['Datetime'].dt.month
df['day_of_month'] = df['Datetime'].dt.day

# Drop the datetime column and use the rest
X = df[['summed_RealPower', 'hour', 'day_of_week', 'month', 'day_of_month']].values
y = df.drop(['Datetime', 'summed_RealPower'], axis=1).values  # Targets: All buildings' power consumption

input_columns = ['summed_RealPower', 'hour', 'day_of_week', 'day_of_month', 'month']
# Define output columns (all buildings)
output_columns = df.columns.difference(['Datetime', 'summed_RealPower', 'hour', 'day_of_week', 'day_of_month', 'month'])

# 2. Reshape input data for LSTM (samples, time steps, features)
X = np.reshape(X, (X.shape[0], 1, X.shape[1]))

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# 4. Building the LSTM Model
model = Sequential()

# LSTM layer with 50 units
model.add(LSTM(units=50, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False))

# Dense layer to output power consumption for each building
model.add(Dense(units=y_train.shape[1]))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# 5. Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_test, y_test), verbose=1)

# 6. Evaluate the model and make predictions
y_pred = model.predict(X_test)

building_names = output_columns 
# Calculate R² score for each building
for i, building in enumerate(building_names):
    r2 = r2_score(y_test[:, i], y_pred[:, i])
    print(f"R² score for Building {i + 1}: {r2:.3f}")

    # Plotting the predictions for each building
    plt.figure(figsize=(10, 6))
    plt.plot(y_test[:100, i], label="Actual Power", color="blue")
    plt.plot(y_pred[:100, i], label="Predicted Power", color="orange", alpha=0.7)
    plt.title(f"Power Consumption Prediction for {building} (R² = {r2:.3f})")
    plt.xlabel("Time Index")
    plt.ylabel("Power Consumption")
    plt.legend()

    # Save the plot as an image file
    plt.savefig(f"./department_power_estimation/LSTM_plots/{building}_power_prediction_r2_{r2:.3f}.png")
    plt.close()

# Optionally: Plot the loss curve
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss during Training')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
