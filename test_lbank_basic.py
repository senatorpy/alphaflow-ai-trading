# File: test_lbank_basic.py
import ccxt

print("🔍 Testing basic LBank connectivity with CCXT (no keys required)...")
try:
    # Create an instance WITHOUT API keys for a basic connectivity test
    exchange = ccxt.lbank() # Use the ID you confirmed is correct

    # Load markets (this tests the basic API connection)
    print("🌐 Loading markets...")
    exchange.load_markets()
    print("✅ Markets loaded successfully.")
    
    # Check if BTC/USDT exists
    symbol = 'BTC/USDT'
    if symbol in exchange.markets:
        print(f"✅ Trading pair {symbol} found.")
    else:
        print(f"⚠️  Trading pair {symbol} NOT found. Available symbols might use different naming (e.g., 'BTC_USDT').")
        # Print first few symbols to check format
        print(f"   First 5 symbols: {list(exchange.markets.keys())[:5]}")

    # Fetch a simple ticker
    print(f"💰 Fetching ticker for {symbol}...")
    ticker = exchange.fetch_ticker(symbol)
    print(f"✅ Ticker fetched. Last price: {ticker['last']}")

except ccxt.NetworkError as e:
    print(f"🌐 Network error during basic test: {type(e).__name__} - {e}")
except ccxt.ExchangeError as e:
    print(f"🔄 Exchange error during basic test: {type(e).__name__} - {e}")
except Exception as e:
    print(f"💥 An unexpected error occurred during basic test: {type(e).__name__} - {e}")

print("✅ Basic LBank test completed.")