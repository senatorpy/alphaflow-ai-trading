# File: check_exchanges.py
import ccxt

print("🔍 Searching for LBank-related exchanges in CCXT...")
lbank_exchanges = [e for e in ccxt.exchanges if 'lbank' in e.lower()]
print(f"🔎 Found potential LBank exchanges: {lbank_exchanges}")

print("\n📋 All available exchanges in CCXT:")
# Print all exchanges for reference (this will be a long list)
# print(ccxt.exchanges)

# Try to instantiate the first one found to see if it works
if lbank_exchanges:
    exchange_id = lbank_exchanges[0]
    print(f"\n🔌 Attempting to create exchange instance for '{exchange_id}'...")
    try:
        exchange_class = getattr(ccxt, exchange_id)
        exchange = exchange_class() # Create without keys for a simple test
        print(f"✅ Successfully created exchange instance: {exchange.id}")
        print("   This is the correct ID to use.")
    except Exception as e:
        print(f"❌ Failed to create instance for '{exchange_id}': {e}")
else:
    print("❓ LBank exchange not found in CCXT list. Check ccxt version or exchange support.")
