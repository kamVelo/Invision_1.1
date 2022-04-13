from time import sleep, perf_counter
from ibapi.wrapper import EWrapper
from ibapi.client import EClient
from ibapi.contract import Contract
from ibapi.order import Order
import logging
from threading import Thread
import atexit
from position import Position
import finnhub
import os
from keys import Keys
class IB(EClient,EWrapper):
    def __init__(self):
        key = Keys.get("finnhub")
        self.finnhub_client = finnhub.Client(api_key=key)
        self.pnlReqId = 1
        self.profit_requested = None
        self.positionReceived = False
        self.orderMade = False
        self.raw_pos = []
        self.balance = 10
        # necessary instantiations per IBAPI documentation
        EWrapper.__init__(self)
        EClient.__init__(self, wrapper=self)

        # connects app to TWS
        self.connect("127.0.0.1", 7497, 999)

        # creates and starts EReader thread
        run = Thread(target=self.run, daemon=True)
        run.start()
        # ensures that app always disconnects cleanly
        atexit.register(self.end)

    def nextValidId(self, orderId: int):
        """
        EWrapper function to receive the new order id and set member variable
        :param orderId: next orderId received from TWS
        :return: None
        """

        super().nextValidId(orderId)
        logging.debug("setting nextValidOrderId: %d", orderId)
        self.nextValidOrderId = orderId

    def error(self, reqId:int, errorCode:int, errorString:str):
        """

        :param reqId: error id (-1 indicates notification not error)
        :param errorCode: error code string
        :param errorString: the actual error description
        :return:
        """
        if reqId != -1 or errorCode not in [2168,2169,2104,2106,2158]: # i.e if not a notification
            print(f"ERROR:: Code:{errorCode} - {errorString}")
        elif errorCode == 201:
            self.insuff_funds = True
        elif errorCode == 103:
            self.nextValidOrderId += 1
    def order(self, instrument, direction):
        """
        function to allow market orders to be created and placed
        :param instrument: the instrument's symbol
        :param direction: i.e BUY or SELL
        :return:  Pos/None for order PLACED (not necessarily successful just placed)
        """
        if len(self.getPositions()) > 0:
            print(self.getPositions())
            return None
        quantity = int((self.getBalance()) / self.getPrice(instrument))
        contract = Contract()
        symbol = instrument
        if len(instrument) == 6: # if it is a forex pair
            sec_type = "CASH"
            symbol = instrument.upper()[0:3]
            currency = instrument.upper()[3:6]

            exchange = "IDEALPRO"
        elif len(instrument) <= 5: # if it is a stock
            sec_type = "STK"
            currency = "USD"
            contract.primaryExchange = "ISLAND"
            exchange = "SMART"
        else: # if it is neither stock or forex pair refuse to place order, return False for failed order.
            return None
        # creates Contract object and fills necessary data

        contract.symbol = symbol
        contract.secType = sec_type
        contract.currency = currency
        contract.exchange = exchange

        # creates order and fills out details
        order = Order()
        order.action = direction.upper()
        order.orderType = "MKT"
        order.totalQuantity = quantity
        # gets latest order id
        old_val = self.nextValidOrderId
        self.reqIds(-1)
        while self.nextValidOrderId == old_val and self.orderMade:  # waits until order id updated.
            pass
        # places the order and returns True since no errors would have been raised by this point.
        self.insuff_funds = None
        self.placeOrder(self.nextValidOrderId, contract, order)
        sleep(0.5)
        if self.insuff_funds: return None
        self.orderMade = True
        self.raw_pos = []
        sleep(5)
        id = None
        count = 0
        while id == None and count<=50: # count prevents the messages per second being sent from being over 50 which closes the connection
            self.reqPositions()
            while len(self.raw_pos) == 0:
                pass
            for pos in self.raw_pos:
                if pos["symbol"] == instrument:
                    id = pos["contract"].conId
            count += 1
        pos = Position(instrument, direction, self)
        pos.opened = True
        pos.open_price = self.getOpenPrice(instrument)
        pos.margin = self.getMargin(pos.symbol)
        pos.shares = quantity
        pos.posId = id
        print(f"Opening Position: %s %s @ %s" % (direction, quantity, pos.open_price))
        return pos

    def accountSummary(self, reqId:int, account:str, tag:str, value:str,currency:str):
        """
        receives the account summary from TWS
        :param reqId: id supplied by reqAccountSummary
        :param account: if multiple accounts then the account selected
        :param tag: tag requested by EClient
        :param value: value returned from request
        :param currency: currency that value is in.
        :return: None
        """
        self.balance = float(value)
    def accountSummaryEnd(self, reqId:int):
        pass

    def getBalance(self):
        """
        gets the balance from TWS
        :return: float balance of total cash value in account
        """
        self.balance = 0
        self.reqAccountSummary(-1, "All","NetLiquidation")
        origin = perf_counter()
        while self.balance == 0:
            diff = perf_counter()-origin
            if diff > 5.0:
                print("Application Error:")
                print("Application not returning balance.")
                self.end()
        self.cancelAccountSummary(-1)
        return self.balance

    def position(self, account: str, contract: Contract, position: float, avgCost: float):
        """
        EWrapper method. called when EClient.reqPositions() or when position opened.
        :param account: account order made with
        :param contract: contract of this position
        :param position: number of shares
        :param avgCost: cost trade made at
        :return: None
        """
        pos = {}
        if contract.secType == 'STK':
            inst_symbol = contract.symbol
        elif contract.secType == 'CASH':
            inst_symbol = contract.symbol+contract.currency
        if abs(position) != 0: # if the position has 0 shares this indicates that it is an old position and not currently open.
            direction = ("SELL", "BUY")[position > 0] # if position is a -ve number then it must be short
            pos["symbol"] = inst_symbol
            pos["direction"] = direction
            pos["no. shares"] = abs(position)
            pos["margin"] = abs(position) * avgCost
            pos["open price"] = avgCost
            pos["contract"] = contract
            self.raw_pos.append(pos)
    def positionEnd(self):
        """
        called when all the position information has been sent.
        :return: None
        """
        self.positionReceived = True

    def getMargin(self, symbol):
        """
        function to get the margin of any given position using EClient.reqPositions()
        :param symbol: symbol of the stock whose margin is being requested
        :return: float margin of the stock.
        """
        self.raw_pos = []
        self.positionReceived = False
        self.reqPositions() # use eclient method to get position info.
        while len(self.raw_pos) == 0: # waits for the position information to be fully received using positionEnd()
            pass
        for pos in self.raw_pos:
            if pos["symbol"] == symbol:
                return pos["margin"] # if the position is found return this.
        return None # if no position is found None will be returned
    def getOpenPrice(self, symbol):
        self.raw_pos = []
        self.reqPositions()
        while len(self.raw_pos) == 0:
            pass
        for pos in self.raw_pos:
            if pos["symbol"] == symbol:
                return pos["open price"]
        return None
    def getPrice(self,symbol):
        return self.finnhub_client.quote(symbol)["c"]
    def pnlSingle(self, reqId: int, pos: int, dailyPnL: float, unrealizedPnL: float, realizedPnL: float, value: float):
        self.profit_requested = unrealizedPnL # this variable will contain the profit requested most recently.

    def getProfit(self,position:Position):
        """
        gets the profit for position requested
        :param position: Position object which is being requested for
        :return: float(profit_requested)
        """
        self.profit_requested = None
        self.reqPnLSingle(self.pnlReqId, "DU3919760", "", position.posId)
        while self.profit_requested == None:
            pass
        self.cancelPnLSingle(self.pnlReqId)
        self.pnlReqId += 1
        return self.profit_requested


    def closePosition(self, position:Position):
        direction = ("BUY","SELL")[position.direction == "BUY"]
        contract = Contract()
        symbol = position.symbol
        if len(position.symbol) == 6:  # if it is a forex pair
            sec_type = "CASH"
            symbol = position.symbol.upper()[0:3]
            currency = position.symbol.upper()[3:6]

            exchange = "IDEALPRO"
        elif len(position.symbol) <= 5:  # if it is a stock
            sec_type = "STK"
            currency = "USD"
            contract.primaryExchange = "ISLAND"
            exchange = "SMART"
        else:  # if it is neither stock or forex pair refuse to place order, return False for failed order.
            return False
        # creates Contract object and fills necessary data

        contract.symbol = symbol
        contract.secType = sec_type
        contract.currency = currency
        contract.exchange = exchange

        # creates order and fills out details
        order = Order()
        order.action = direction.upper()
        order.orderType = "MKT"
        order.totalQuantity = position.shares

        # gets latest order id
        old_val = self.nextValidOrderId
        self.reqIds(-1)
        while self.nextValidOrderId == old_val and self.orderMade:  # waits until order id updated.
            pass
        # places the order and returns True since no errors would have been raised by this point.
        self.placeOrder(self.nextValidOrderId, contract, order)
        return True
    def getPositions(self):
        self.raw_pos = []
        self.positionReceived = False
        self.reqPositions()
        while not self.positionReceived: # waits until list is populated by (EWrapper.position)
            pass
        return self.raw_pos

    def end(self):
        print("TRADER DISCONNECTING")
        self.done = True
        self.disconnect()
        os._exit(0)
if __name__ == "__main__":
    m = IB()
