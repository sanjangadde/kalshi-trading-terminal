import os
from dotenv import load_dotenv
from cryptography.hazmat.primitives import serialization
import asyncio
import time
from collections import defaultdict

from clients import KalshiHttpClient, KalshiWebSocketClient, Environment

# Load environment variables
load_dotenv()
env = Environment.PROD  # toggle environment here
KEYID = os.getenv('DEMO_KEYID') if env == Environment.DEMO else os.getenv('PROD_KEYID')
KEYFILE = os.getenv('DEMO_KEYFILE') if env == Environment.DEMO else os.getenv('PROD_KEYFILE')

try:
    with open(KEYFILE, "rb") as key_file:
        private_key = serialization.load_pem_private_key(
            key_file.read(),
            password=None
        )
except FileNotFoundError:
    raise FileNotFoundError(f"Private key file not found at {KEYFILE}")
except Exception as e:
    raise Exception(f"Error loading private key: {str(e)}")

# Initialize the HTTP client
client = KalshiHttpClient(
    key_id=KEYID,
    private_key=private_key,
    environment=env
)

# Get account balance
balance = client.get_balance()
print("Balance:", balance)

# ---- New: find highest-volume market over last 24h ----
# ---- New: find highest-volume market over last 24h ----
now_sec = int(time.time())          # seconds, not ms
day_sec = 24 * 60 * 60
min_ts = now_sec - day_sec

vol_by_ticker = defaultdict(int)
cursor = None
page = 0
'''
while True:
    # Now pass seconds, not ms
    resp = client.get_markets()
    trades = resp.get("trades", [])
    if not trades:
        break

    for t in trades:
        ticker = t.get("ticker")
        qty = t.get("quantity", t.get("size", 0))
        try:
            vol_by_ticker[ticker] += int(qty)
        except Exception:
            pass

    cursor = resp.get("cursor")
    page += 1
    if not cursor:
        break

if not vol_by_ticker:
    raise RuntimeError("No trades found in the last 24 hours; cannot select a market.")'''
top_ticker = 'KXFEDDECISION-25SEP-C25'
print(f"Top 24h volume market: {top_ticker}")

# Initialize the WebSocket client for only that market
ws_client = KalshiWebSocketClient(
    key_id=KEYID,
    private_key=private_key,
    environment=env,
    market_tickers=[top_ticker],  # subscribe only to this ticker
)

# Connect via WebSocket
asyncio.run(ws_client.connect())
