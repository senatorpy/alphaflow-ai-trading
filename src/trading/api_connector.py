# File: src/trading/api_connector.py
"""
Handles connection and data fetching from the LBank API using the CCXT library.
This module defines the LBankConnector class for interacting with LBank's market data
and simulating trades.
"""

import ccxt
import pandas as pd
import os
# Import settings from our config module
# Make sure config/settings.py defines LBANK_API_KEY and LBANK_SECRET_KEY
from config.settings import LBANK_API_KEY, LBANK_SECRET_KEY

class LBankConnector:
    """
    A connector class for interacting with the LBank cryptocurrency exchange.
    
    Attributes:
        exchange (ccxt.lbank2): An instance of the CCXT LBank v2 exchange class,
                                configured with API keys.
    """

    def __init__(self):
        """
        Initializes the LBank connector with API credentials.
        Raises a ValueError if API keys are not found in config.settings.
        """
        if not LBANK_API_KEY or not LBANK_SECRET_KEY:
            raise ValueError("❌ LBank API keys not found. Please set LBANK_API_KEY and LBANK_SECRET_KEY in config/settings.py")

        # Initialize the CCXT exchange instance for LBank v2
        # Note: ccxt uses 'lbank2' for the current API version
        self.exchange = ccxt.lbank({
            'apiKey': LBANK_API_KEY,
            'secret': LBANK_SECRET_KEY,
            # Enables built-in rate limiting to avoid exceeding API limits
            'enableRateLimit': True,
            # Optional: Set a timeout for API requests
            'timeout': 30000, # 30 seconds
        })
        print("🔗 Initialized connection to LBank API.")

    def fetch_historical_data(self, symbol='BTC/USDT', timeframe='1h', limit=1000):
        """
        Fetches historical OHLCV (Open, High, Low, Close, Volume) data.
        
        Args:
            symbol (str): The trading pair symbol (e.g., 'BTC/USDT', 'ETH/USDT').
            timeframe (str): The granularity of the data (e.g., '1m', '5m', '1h', '1d').
            limit (int): The maximum number of data points to fetch.
            
        Returns:
            pd.DataFrame: A DataFrame with a DatetimeIndex and columns ['open', 'high', 'low', 'close', 'volume'].
                          Returns None if an error occurs.
        """
        try:
            print(f"📊 Fetching historical data for {symbol} ({timeframe}) from LBank...")
            # Use CCXT's standard fetch_ohlcv method
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
            
            # Check if data was returned
            if not ohlcv:
                print(f"⚠️  No historical data found for {symbol}.")
                return None

            # Convert to Pandas DataFrame for easier handling
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            # Convert timestamp from milliseconds to pandas datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            # Set timestamp as the DataFrame index
            df.set_index('timestamp', inplace=True)
            
            print(f"✅ Successfully fetched {len(df)} data points.")
            return df

        except ccxt.NetworkError as e:
            print(f"🌐 Network error while fetching data: {e}")
        except ccxt.ExchangeError as e:
            print(f"🔄 Exchange error while fetching data: {e}")
        except Exception as e:
            print(f"💥 An unexpected error occurred while fetching data: {e}")
        
        return None # Return None if any error happens

    def get_current_price(self, symbol='BTC/USDT'):
        """
        Fetches the current ticker price for a symbol.
        
        Args:
            symbol (str): The trading pair symbol (e.g., 'BTC/USDT').
            
        Returns:
            float: The last traded price, or None if an error occurs.
        """
        try:
            print(f"💰 Fetching current price for {symbol}...")
            # Use CCXT's standard fetch_ticker method
            ticker = self.exchange.fetch_ticker(symbol)
            price = ticker['last']
            print(f"✅ Current price for {symbol}: {price}")
            return price

        except ccxt.NetworkError as e:
            print(f"🌐 Network error while fetching price: {e}")
        except ccxt.ExchangeError as e:
            print(f"🔄 Exchange error while fetching price: {e}")
        except Exception as e:
            print(f"💥 An unexpected error occurred while fetching price: {e}")
        
        return None # Return None if any error happens

    def execute_dry_run_trade(self, symbol, side, amount, price=None):
        """
        Simulates a trade execution (dry run) for logging/debugging purposes.
        In a real scenario, this would place an order using self.exchange.create_order().
        
        Args:
            symbol (str): The trading pair symbol (e.g., 'BTC/USDT').
            side (str): Order side ('buy' or 'sell').
            amount (float): The amount of the base currency (e.g., BTC) to trade.
            price (float, optional): The limit price. If None, a market order is simulated.
        """
        # Determine order type for display
        order_type = "LIMIT" if price else "MARKET"
        price_info = f" @ {price} USDT" if price else " (Market Order)"

        # Use the current market price if not provided for simulation
        if price is None:
            price = self.get_current_price(symbol)
            if price is None:
                print(f"❌ Could not fetch current price for {symbol}. Dry-run trade not logged.")
                return

        print(f"🚨 DRY RUN TRADE: {side.upper()} {amount:.6f} {symbol.split('/')[0]} {order_type}{price_info}")
        print(f"   - Estimated Cost/Revenue: {amount * price:.2f} {symbol.split('/')[1]}")
        print("   - No actual order was placed on LBank.")


# --- Example Usage (for local testing) ---
# This block runs only if the script is executed directly.
if __name__ == '__main__':
    print("🧪 Testing LBankConnector...")
    try:
        # 1. Initialize the connector
        connector = LBankConnector()

        # 2. Fetch current price
        current_price = connector.get_current_price('BTC/USDT')
        if current_price:
            print(f"   - Current BTC/USDT Price: ${current_price}")

        # 3. Fetch recent historical data
        print("\n--- Fetching Historical Data ---")
        data = connector.fetch_historical_data('BTC/USDT', '1h', limit=10)
        if data is not None and not data.empty:
            print(data.tail()) # Print the last few rows

        # 4. Simulate a dry-run trade
        print("\n--- Simulating Dry-Run Trade ---")
        if current_price:
            connector.execute_dry_run_trade('BTC/USDT', 'buy', 0.001, price=current_price)

    except ValueError as e:
        # This handles the error raised in __init__ if keys are missing
        print(e)
    except Exception as e:
        print(f"💥 An error occurred during testing: {e}")

    print("✅ LBankConnector test completed.")
 
