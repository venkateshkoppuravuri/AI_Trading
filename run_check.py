import sys
sys.path.insert(0, 'C:/Users/laptop/OneDrive/Desktop/AI Trading')
from trading.client import AlpacaClient
c = AlpacaClient()
pos = c.get_position('AAPL')
print('AAPL position:', pos)
if pos:
    qty = float(pos.get('qty', 0))
    print(f'qty = {qty}')
    if qty < 0:
        order = c.place_market_order('AAPL', int(abs(qty)), 'buy')
        print(f'Closed AAPL short — order id: {order.get("id")}')
    else:
        print('AAPL is a long position, no action needed')
else:
    print('No AAPL position found')
