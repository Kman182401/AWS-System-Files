from ib_insync import IB, util
from datetime import datetime

def main(host: str = '127.0.0.1', port: int = 4002, client_id: int = 101):
    ib = IB()
    try:
        ib.connect(host, port, clientId=client_id)
        print('Server time:', ib.reqCurrentTime())
        print('Positions:', ib.positions())
        print('Open orders:', ib.openOrders())
        print('Completed orders:', ib.reqCompletedOrders(apiOnly=True))
    finally:
        ib.disconnect()

if __name__ == '__main__':
    main()
