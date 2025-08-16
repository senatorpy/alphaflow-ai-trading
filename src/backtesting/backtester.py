# File: src/backtesting/backtester.py
"""
Backtests a trading strategy using historical data and a trained model.
"""

# --- Add project root to path for direct script execution ---
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
# --- End of path modification ---

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd # <-- Ensure this is present
# Import our model and connector
# Make sure these imports match the class names in their respective files
from src.trading.model import AdaptiveTradingModel # <-- ADD THIS LINE
from src.trading.api_connector import LBankConnector # Or BinanceConnector

class TradingBacktester:
    """
    Simulates trading performance using a trained model and historical data.
    
    Attributes:
        model (AdaptiveTradingModel): The trained AI model for predictions.
        historical_data (pd.DataFrame): DataFrame containing historical price data.
        price_data (np.ndarray): The 'close' prices extracted from historical_data.
        timestamps (pd.DatetimeIndex): The timestamps corresponding to price_data.
    """

    def __init__(self, model: AdaptiveTradingModel, historical_data: pd.DataFrame):
        """
        Initializes the backtester with a model and data.
        
        Args:
            model (AdaptiveTradingModel): A trained instance of the prediction model.
            historical_data (pd.DataFrame): DataFrame with a DatetimeIndex and a 'close' column.
        """
        if 'close' not in historical_data.columns:
            raise ValueError("Historical data must contain a 'close' column.")
        
        self.model = model
        self.historical_data = historical_data
        self.price_data = historical_data['close'].values
        self.timestamps = historical_data.index
        print("🏁 Initialized TradingBacktester.")

    def run(self, initial_balance=10000.0, trade_amount_percent=0.1, fee_percent=0.1):
        """
        Executes the backtesting simulation.
        
        Args:
            initial_balance (float): The starting account balance (e.g., in USDT).
            trade_amount_percent (float): Percentage of balance/position to trade each time (0.0 to 1.0).
            fee_percent (float): Transaction fee percentage (e.g., 0.1 for 0.1%).
            
        Returns:
            tuple: A tuple containing (list of trades, final total return percentage).
        """
        balance = initial_balance
        position = 0.0  # Amount of cryptocurrency held
        trades = []  # List to store trade details
        portfolio_values = [initial_balance]  # Track portfolio value over time
        fee_multiplier = 1 - (fee_percent / 100.0) # e.g., 0.999 for 0.1% fee

        lookback = self.model.lookback_window
        print(f"🚀 Starting backtest simulation with ${initial_balance:.2f}...")
        print(f"   - Lookback Window: {lookback}")
        print(f"   - Trade Amount: {trade_amount_percent*100:.1f}% of balance/position")
        print(f"   - Fee: {fee_percent}% per trade")

        # Iterate through the data, leaving enough points for lookback and one prediction
        for i in range(lookback, len(self.price_data)):
            # 1. Get the last 'lookback_window' prices for the model
            recent_prices = self.price_data[i-lookback:i]
            current_price = self.price_data[i]
            
            # 2. Get model prediction for the next price
            predicted_price = self.model.predict(recent_prices)
            
            # 3. Define simple trading strategy based on prediction
            # Strategy: Buy if prediction is significantly higher, Sell if significantly lower.
            # The thresholds (1.01, 0.99) can be adjusted as hyperparameters.
            action = None
            amount = 0.0
            
            # --- Buy Signal ---
            if predicted_price > current_price * 1.01 and balance > current_price:
                # Calculate amount to buy based on percentage of current balance
                amount = (balance * trade_amount_percent) / current_price
                cost = amount * current_price
                fee = cost * (1 - fee_multiplier)
                balance -= (cost + fee) # Deduct cost and fee
                position += amount
                action = 'BUY'
            
            # --- Sell Signal ---
            elif predicted_price < current_price * 0.99 and position > 0:
                 # Calculate amount to sell based on percentage of current position
                amount = position * trade_amount_percent
                revenue = amount * current_price
                fee = revenue * (1 - fee_multiplier)
                balance += (revenue - fee) # Add revenue, deduct fee
                position -= amount
                action = 'SELL'
            
            # 4. Log trade details if an action was taken
            if action:
                trades.append({
                    'timestamp': self.timestamps[i],
                    'action': action,
                    'price': current_price,
                    'amount': amount,
                    'balance_after': balance,
                    'position_after': position,
                    'predicted_price': predicted_price
                })
                # Optional: Print trade for verbose output during backtest
                # print(f"   [{self.timestamps[i].strftime('%Y-%m-%d %H:%M')}] {action} {amount:.6f} at {current_price:.2f}")

            # 5. Calculate and store portfolio value for plotting
            portfolio_value = balance + (position * current_price)
            portfolio_values.append(portfolio_value)

        # --- Final Calculation ---
        # Calculate value using the very last known price
        final_price = self.price_data[-1] 
        final_portfolio_value = balance + (position * final_price)
        total_return_percent = ((final_portfolio_value - initial_balance) / initial_balance) * 100
        
        print(f"✅ Backtest simulation completed.")
        print(f"💰 Final Portfolio Value: ${final_portfolio_value:.2f}")
        print(f"📊 Total Return: {total_return_percent:.2f}%")

        # --- Performance Metrics ---
        # Calculate Buy & Hold performance for comparison
        buy_hold_return = ((self.price_data[-1] - self.price_data[lookback]) / self.price_data[lookback]) * 100
        print(f"📈 Buy & Hold Return (for comparison): {buy_hold_return:.2f}%")

        # --- Plotting Results ---
        self._plot_results(portfolio_values, initial_balance, trades)

        return trades, total_return_percent

    def _plot_results(self, portfolio_values, initial_balance, trades):
        """
        Plots the portfolio value and underlying asset price over time.
        """
        # Data for plotting
        plot_timestamps = self.timestamps[self.model.lookback_window:] # Align with portfolio values
        asset_prices = self.price_data[self.model.lookback_window:]

        fig, ax1 = plt.subplots(figsize=(14, 7))

        color = 'tab:blue'
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Portfolio Value (USDT)', color=color)
        ax1.plot(plot_timestamps, portfolio_values[1:], label='AI Strategy Portfolio', color=color, linewidth=2)
        ax1.axhline(y=initial_balance, color='r', linestyle='--', alpha=0.7, label=f'Initial Balance (${initial_balance:.0f})')
        ax1.tick_params(axis='y', labelcolor=color)

        # Formatting x-axis dates
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator())
        fig.autofmt_xdate() # Rotate date labels

        # Create a second y-axis for the asset price
        ax2 = ax1.twinx()
        color = 'tab:orange'
        ax2.set_ylabel('BTC/USDT Price', color=color)
        ax2.plot(plot_timestamps, asset_prices, label='BTC/USDT Price', color=color, alpha=0.7)
        ax2.tick_params(axis='y', labelcolor=color)

        # Title and Legend
        fig.suptitle('Backtest Results: AI Strategy vs. BTC Price', fontsize=16)
        # Combine legends from both axes
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='upper left')

        ax1.grid(True, alpha=0.3)
        fig.tight_layout()
        
        # Save the plot
        plot_filename = 'backtest_results.png'
        plt.savefig(plot_filename)
        print(f"📊 Backtest plot saved as '{plot_filename}'.")
        
        # Optionally, display the plot (might block script execution)
        # plt.show() 
        plt.close(fig) # Close to free memory


# --- Example Usage (for local testing) ---
# This block runs only if the script is executed directly.
if __name__ == '__main__':
    print("🧪 Running a quick backtest using Binance data and a simple model...")
    try:
        # 1. Initialize Binance connector
        print("--- Initializing Binance Connector ---")
        connector = LBankConnector()

        # 2. Fetch a small amount of recent historical data (last 500 hourly candles)
        print("\n--- Fetching Historical Data ---")
        # Using a common trading pair and timeframe
        symbol = 'BTC/USDT'
        timeframe = '1h'
        limit = 500
        data = connector.fetch_historical_data(symbol, timeframe, limit)
        
        if data is None or data.empty:
            print("❌ Failed to fetch data. Exiting backtest.")
            exit(1)

        # 3. Initialize the AI model with a smaller lookback for quicker testing
        print("\n--- Initializing AI Model ---")
        lookback_window = 30
        model = AdaptiveTradingModel(lookback_window=lookback_window)

        # 4. Train the model on the fetched data (use fewer epochs for speed)
        print("\n--- Training AI Model ---")
        model.train(data['close'].values, epochs=5) # Very short training for demo

        # 5. Initialize the Backtester
        print("\n--- Initializing Backtester ---")
        backtester = TradingBacktester(model, data)

        # 6. Run the backtest simulation
        print("\n--- Running Backtest ---")
        trades, total_return = backtester.run(initial_balance=1000.0, trade_amount_percent=0.1, fee_percent=0.1)
        
        print(f"\n✅ Quick backtest finished. Final Return: {total_return:.2f}%")
        # Print first and last trade for inspection
        if trades:
            print(f"   - First Trade: {trades[0]['timestamp']} - {trades[0]['action']} {trades[0]['amount']:.6f}")
            print(f"   - Last Trade: {trades[-1]['timestamp']} - {trades[-1]['action']} {trades[-1]['amount']:.6f}")

    except Exception as e:
        print(f"💥 An error occurred during the quick backtest: {e}")
        import traceback
        traceback.print_exc()

    print("✅ Quick backtest example completed.")
 
