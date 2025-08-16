# File: src/trading/model.py
"""
LSTM-based model for predicting cryptocurrency price movements.
This module defines the AdaptiveTradingModel class which encapsulates
the data preprocessing, model creation, training, and prediction logic.
"""

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

class AdaptiveTradingModel:
    """
    An LSTM neural network model for forecasting price trends.
    
    Attributes:
        lookback_window (int): Number of previous time steps to use as input.
        scaler (MinMaxScaler): Scaler to normalize price data.
        model (tf.keras.Sequential): The trained Keras LSTM model.
    """

    def __init__(self, lookback_window=60):
        """
        Initializes the model with a specified lookback window.
        
        Args:
            lookback_window (int): The number of past data points the model uses
                                   to make a prediction. Default is 60.
        """
        self.lookback_window = lookback_window
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = self._build_model()
        print(f"🧠 Initialized AdaptiveTradingModel with lookback window: {lookback_window}")

    def _build_model(self):
        """
        Constructs the LSTM neural network architecture.
        
        Returns:
            tf.keras.Sequential: The untrained Keras model.
        """
        model = tf.keras.Sequential([
            # First LSTM layer with return_sequences=True for stacking
            tf.keras.layers.LSTM(50, return_sequences=True, input_shape=(self.lookback_window, 1)),
            # Dropout for regularization to prevent overfitting
            tf.keras.layers.Dropout(0.2),
            
            # Second LSTM layer
            tf.keras.layers.LSTM(50, return_sequences=False),
            # Another Dropout layer
            tf.keras.layers.Dropout(0.2),
            
            # Dense layers for final prediction
            tf.keras.layers.Dense(25, activation='relu'), # Intermediate layer
            tf.keras.layers.Dense(1) # Output layer for single price prediction
        ])
        
        # Compile the model with optimizer and loss function
        model.compile(optimizer='adam', loss='mean_squared_error')
        print("🏗️  Built LSTM model architecture.")
        return model

    def train(self, data, epochs=20, batch_size=32):
        """
        Trains the LSTM model on historical price data.
        
        Args:
            data (np.ndarray): 1D Numpy array of historical prices (e.g., closing prices).
            epochs (int): Number of times the model will see the entire dataset. Default is 20.
            batch_size (int): Number of samples per gradient update. Default is 32.
        """
        print("📈 Starting model training...")
        # 1. Scale the data to be between 0 and 1
        scaled_data = self.scaler.fit_transform(data.reshape(-1, 1))
        print(f"   - Data scaled. Shape: {scaled_data.shape}")

        # 2. Prepare training data (X_train: sequences, y_train: next value)
        X_train, y_train = [], []
        for i in range(self.lookback_window, len(scaled_data)):
            # Take the last 'lookback_window' values as input
            X_train.append(scaled_data[i-self.lookback_window:i, 0])
            # The next value is the target
            y_train.append(scaled_data[i, 0])
        
        # 3. Convert lists to NumPy arrays for Keras
        X_train, y_train = np.array(X_train), np.array(y_train)
        print(f"   - Training data prepared. X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

        # 4. Reshape X_train to be 3D for LSTM: [samples, time steps, features]
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        print(f"   - X_train reshaped for LSTM. New shape: {X_train.shape}")

        # 5. Train the model
        history = self.model.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)
        print("✅ Model training completed successfully.")
        # Optionally, you could return the history for plotting loss
        # return history 

    def predict(self, recent_data):
        """
        Predicts the next price based on the most recent data points.
        
        Args:
            recent_data (np.ndarray): 1D Numpy array of the most recent 'lookback_window' prices.
            
        Returns:
            float: The predicted next price (in the original scale).
        """
        # 1. Ensure the input data has the correct length
        if len(recent_data) != self.lookback_window:
            raise ValueError(f"Input data must have length {self.lookback_window}, got {len(recent_data)}")

        # 2. Scale the recent data using the scaler fitted during training
        scaled_data = self.scaler.transform(recent_data.reshape(-1, 1))
        
        # 3. Prepare the input for the model (1 sample, lookback_window time steps, 1 feature)
        X_test = np.array([scaled_data[:, 0]]) # Take the scaled values
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        print(f"🔮 Predicting with input shape: {X_test.shape}")

        # 4. Make the prediction
        prediction_scaled = self.model.predict(X_test, verbose=0) # verbose=0 for cleaner output
        
        # 5. Inverse transform the prediction to get the actual price value
        prediction = self.scaler.inverse_transform(prediction_scaled)
        
        return prediction[0][0] # Return the single predicted value

# Example usage (for local testing, can be removed or commented out later)
if __name__ == '__main__':
    print("🧪 Running a simple test of the AdaptiveTradingModel...")
    # 1. Create some dummy data (e.g., a sine wave)
    dummy_data = np.sin(np.linspace(0, 10*np.pi, 1000)) * 100 + 50 # Prices between 0 and 100
    print(f"   - Created dummy data of {len(dummy_data)} points.")

    # 2. Initialize the model
    model = AdaptiveTradingModel(lookback_window=30)

    # 3. Train the model on the dummy data (very briefly)
    model.train(dummy_data, epochs=3, batch_size=16) # Short training for demo

    # 4. Make a prediction using the last 'lookback_window' points
    last_prices = dummy_data[-30:]
    predicted_price = model.predict(last_prices)
    actual_next_price = dummy_data[-1] # The actual last price in our dummy data

    print(f"   - Last known price (actual): {actual_next_price:.2f}")
    print(f"   - Predicted next price: {predicted_price:.2f}")
    print("   - Note: Performance on dummy data is not indicative of real market performance.")
    print("✅ Test completed.")
 
