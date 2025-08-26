import asyncio
import base64
import json
import time
import websockets
from cryptography.hazmat.primitives import serialization, hashes
from cryptography.hazmat.primitives.asymmetric import padding
import os
from dotenv import load_dotenv
load_dotenv()

from pipe_util import PipeWriter

# Configuration
KEY_ID = os.getenv("PROD_KEYID")
PRIVATE_KEY_PATH = os.getenv("PROD_KEYFILE")
MARKET_TICKER = 'KXHIGHLAX-25AUG26-B75.5'.upper()  # Replace with any open market
WS_URL = "wss://api.elections.kalshi.com/trade-api/ws/v2"

def sign_pss_text(private_key, text: str) -> str:
    """Sign message using RSA-PSS"""
    message = text.encode('utf-8')
    signature = private_key.sign(
        message,
        padding.PSS(
            mgf=padding.MGF1(hashes.SHA256()),
            salt_length=padding.PSS.DIGEST_LENGTH
        ),
        hashes.SHA256()
    )
    return base64.b64encode(signature).decode('utf-8')

def create_headers(private_key, method: str, path: str) -> dict:
    """Create authentication headers"""
    timestamp = str(int(time.time() * 1000))
    msg_string = timestamp + method + path.split('?')[0]
    signature = sign_pss_text(private_key, msg_string)
    
    return {
        "Content-Type": "application/json",
        "KALSHI-ACCESS-KEY": KEY_ID,
        "KALSHI-ACCESS-SIGNATURE": signature,
        "KALSHI-ACCESS-TIMESTAMP": timestamp,
    }

async def orderbook_websocket():
    """Connect to WebSocket and subscribe to orderbook"""
    # Load private key
    with open(PRIVATE_KEY_PATH, 'rb') as f:
        private_key = serialization.load_pem_private_key(
            f.read(),
            password=None
        )
    pipe = PipeWriter("/tmp/orderbook_pipe", max_wait=2.0)
    # Create WebSocket headers
    ws_headers = create_headers(private_key, "GET", "/trade-api/ws/v2")
    
    async with websockets.connect(WS_URL, additional_headers=ws_headers) as websocket:
        print(f"Connected! Subscribing to orderbook for {MARKET_TICKER}")
        
        # Subscribe to orderbook
        subscribe_msg = {
            "id": 1,
            "cmd": "subscribe",
            "params": {
                "channels": ['ticker', 'orderbook_delta', 'trade'],
                "market_ticker": MARKET_TICKER
            }
        }
        await websocket.send(json.dumps(subscribe_msg))
        
        # Process messages
        async for message in websocket:
            data = json.loads(message)
            msg_type = data.get("type")
            try:
                pipe.write_json(data)
            except Exception as e:
                # Non-fatal: keep streaming even if pipe has issues
                print(f"[pipe] write failed: {e}")
            
            if msg_type == "subscribed":
                print(f"Subscribed: {data}")
                
            elif msg_type == "orderbook_snapshot":
                print(f"Orderbook snapshot: {data}")
                
            elif msg_type == "orderbook_delta":
                # The client_order_id field is optional - only present when you caused the change
                if 'client_order_id' in data.get('data', {}):
                    print(f"Orderbook update (your order {data['data']['client_order_id']}): {data}")
                else:
                    print(f"Orderbook update: {data}")
                        
            elif msg_type == "error":
                print(f"Error: {data}")
            
            elif msg_type == "ticker":
                print(f"Ticker Update: {data}")
            
            elif msg_type == "trade":
                print(f"Trade: {data}")

# Run the example
if __name__ == "__main__":
    asyncio.run(orderbook_websocket())