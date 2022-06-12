

from sys import exit
from time import sleep
from datetime import datetime as dt, time, timedelta
import os
from download import download, getFeature
from string import punctuation as special_chars
from classifier import Classifier
from positionManager import PositionManager
import requests as rq
from ib import IB
from getInstrument import getMovers
from position import Position
import pytz
from reinforcement import Environment as RL
#TODO: everytime you download data for a stock (monthly) add this data to the original data file
#TODO: stocks data folder should have a file which contains metadata specifically abt how much of the data the neural network has already been trained on
#TODO: save neural network so that you don't have to retrain it on the whole dataset everytime - this means that the rest of the dataset will also be "internalised" each time
#TODO: add wifi-down feature that keeps hte application open if there's no internet connection
#TODO: improve tws disconnection management.
class Trader:
    def __init__(self):
        self.beginning = True
        # gets stock to trade
        getset = input("Enter Y to select a specific stock, C to choose from a screener, or N to have it chosen automatically,\nor enter F to automatically choose at next open: ")
        if getset.upper() == "Y":
            self.symbol = input("Enter a  stock: ").strip().upper()
        elif getset.upper() == "C":
            self.symbol = self.getStock()
        elif getset.upper() =="N":
            self.symbol = self.getStock(auto=True)
        elif getset.upper() == "F":
            ny = pytz.timezone("America/New_York")
            getStockTime = dt(2021, 9, 13, 10, 5, tzinfo=ny).time() # get stock at 10:05 new york time. This allows for 15 minute time delay as well as 20 minutes of trading
            dontGet = dt(2021, 9, 13, 16, 30, tzinfo=ny).time()
            timeToGet = lambda: getStockTime <= dt.now().astimezone(ny).time() < dontGet
            while not timeToGet():
                pass
            try:
                self.symbol = self.getStock(auto=True)
            except Exception:
                self.symbol = "MSFT"
        else:
            print("Invalid input. Closing Program.")
            exit(0)

        # checks if stock data is already downloaded or not:
        if not os.path.isdir(os.path.join("data",self.symbol)):
            # if not download data now.
            print("No data downloaded for this symbol.\n Downloading now...")
            download(self.symbol)

        # instantiates classifier
        self.quant = RL(self.symbol, False)


        # instantiates trader
        self.executor = IB()

        # if trader not connected exit app
        if not self.executor.isConnected():
            print("TWS connection error.")
            print("Check if application is open.")
            print("Exiting Application.")
            exit(0)
        # gets the balance for the account
        self.init_bal = self.executor.getBalance()
        # instantiating a position manager
        self.pos_manager = PositionManager(self.symbol)
        # declaring feature matrix to be populated
        self.feature = [[]]

        # declaring position variable holding the current position of the trader
        self.position = None

        #declaring go-variables
        # these variables tell whatStep whether or not a step in the process has just been completed
        # i.e balance is checked at 15:25. when whatStep is called it will tell caller to check balance
        # however the check will end before a minute has passed and subsequently whatStep will continue to
        # tell the caller to check balance until 15:26. by using a flag which is set to false once a step has been
        # completed and waits until after a minute/second has passed (i.e 15:26) whatStep will only tell the caller
        # to do a step once

        self.go_for_bal = True
        self.go_for_check = True
        self.go_for_trade = True
        self.go_for_finish = True

        # failure counters
        # for each step how many failures?
        self.bal_check_fail = 0
        self.pos_check_fail = 0
        self.trade_fail = 0
        self.newDay = False

        if len(self.executor.getPositions()) > 0: # i.e there is a position left open from the day before for whatever reason
            print("Positions are currently open.")
        self.run()
    def run(self):

        print("TRADER RUNNING")
        # welcomes user
        self.welcomeMsg()


        while True:

            # time variables to control when different steps happen
            step = self.whatStep()

            # if its time to check balance
            if step == "CHECK BALANCE":
                successful = self.checkBalance()
                if not successful: self.go_for_bal = True # this means if the check wasn't successful do it again.

            # if it is time to check the position
            if step == "CHECK TRADE":
                successful = self.checkTrade()
                if not successful: self.go_for_trade = True # this will never happen as True is hard-coded but this may change

            # if it is time to trade:
            if step == "TRADE":
                self.trade()

            # if it's time to finish up
            if step == "FINISH UP":
                self.finish()
    def refresh(self):
        self.beginning = True
        # gets stock to trade
        getset = input(
            "Enter Y to select a specific stock, C to choose from a screener, or N to have it chosen automatically: ")
        if getset.upper() == "Y":
            self.symbol = input("Enter a  stock: ").upper()
        elif getset.upper() == "C":
            self.symbol = self.getStock()
        elif getset.upper() == "N":
            self.symbol = self.getStock(auto=True)
        else:
            print("Invalid input. Closing Program.")
            exit(0)

        # checks if stock data is already downloaded or not:
        if not os.path.isdir(self.symbol):
            # if not download data now.
            print("No data downloaded for this symbol.\n Downloading now...")
            download(self.symbol)

        # instantiates classifier
        self.quant = Classifier(self.symbol)
        self.quant.prepNN()

        self.init_bal = self.executor.getBalance()
        # instantiating a position manager
        self.pos_manager = PositionManager(self.symbol)
        # declaring feature matrix to be populated
        self.feature = [[]]

        # declaring position variable holding the current position of the trader
        self.position = None

        # declaring go-variables
        # these variables tell whatStep whether or not a step in the process has just been completed
        # i.e balance is checked at 15:25. when whatStep is called it will tell caller to check balance
        # however the check will end before a minute has passed and subsequently whatStep will continue to
        # tell the caller to check balance until 15:26. by using a flag which is set to false once a step has been
        # completed and waits until after a minute/second has passed (i.e 15:26) whatStep will only tell the caller
        # to do a step once

        self.go_for_bal = True
        self.go_for_check = True
        self.go_for_trade = True
        self.go_for_finish = True

        # failure counters
        # for each step how many failures?
        self.bal_check_fail = 0
        self.pos_check_fail = 0
        self.trade_fail = 0
        self.newDay = False

        if len(
                self.executor.getPositions()) > 0:  # i.e there is a position left open from the day before for whatever reason
            print("Positions are currently open.")
    def trade(self):
        """
        called every hour to do a trade
        this does not mean a position will be opened every hour
        situations where a position would not be opened include:
            - previous position was in the same direction but was closed at a loss.
            - a position is already open in this direction
            - the prediction returned by the Classifier is None (i.e confidence too low).

        :return: None
        """
        # get data
        self.feature = getFeature(self.symbol)

        #get prediction
        pred = self.quant.predict(self.feature)

        if self.position: # if a position exists (including if it wasn't actually opened because disallowed)

            # announce prediction

            print(f"\t{dt.today()} - Prediction: {pred} | Current position: {self.position.direction}")

            if self.position.direction == pred: # if open position in the same direction
                print("Position already open in the same direction")
            elif pred != None: # position is open in opposite direction, or disallowed
                self.position.close()
                self.pos_manager.record(self.position)
                self.position = Position(self.symbol, pred, self.executor)

                # this line is a little weird.
                # first we have created a position object.
                # then we call the Position.open() func, this function returns the output of IB.order
                # the output of IB.order is a Position object with all data filled in.
                # therefore this line
                # - assigns the current position
                # - to the return given by the positions opening using IB
                self.position = self.position.open()

        else: # if no position currently exists
            # announce a prediction
            print(f"\t{dt.today()} - Prediction: {pred}")

            # get last position
            last_pos = self.pos_manager.getLastPosition()

            # if there was a previous position and it was in the same direction and it was a loss
            if last_pos and last_pos.direction == pred and last_pos.status== "LOSS":
                self.position = Position(self.symbol, pred, self.executor)
                self.position = self.position.open(disallow=True)
            else: # if no position is currently open for any reason other than
                self.position = Position(self.symbol, pred, self.executor)
                self.position = self.position.open()




    def checkBalance(self):
        """
        output balance info and check it hasn't fallen too far.
        :return: True/false indicating whether check was successful or not
        """
        successful = True
        cur_bal = self.executor.getBalance()
        try:
            print(f"\t\t{dt.today()} - Current balance: {cur_bal} | Inital balance: {self.init_bal} | Change: {round((cur_bal / self.init_bal * 100) - 100, 2)}%")
            if cur_bal/self.init_bal <= 0.85: # if we've lost 15% or greater balance in this session
                print("BALANCE FALLEN TOO FAR. CLOSING POSITIONS.")
                if self.position: # if a position exists
                    closed = self.position.close() # gets the result of closing
                    if closed == False:
                        print("FAILED TO CLOSE POSITION.")
                        print("EXITING TRADER")
                        exit(0)
                    else:
                        self.pos_manager.record(self.position) # if the position closed successfully. record it.
        except ZeroDivisionError and TypeError and AttributeError: # this will happen if the initial balance wasn't recorded properly.
            print("FAILED BALANCE CHECK")
            if type(cur_bal) == int and cur_bal > 0: # so just set the init_bal now if cur_bal is valid.
                self.init_bal = cur_bal
                successful = False
                self.bal_check_fail += 1
        return successful

    def checkTrade(self):
        """
        checks the current position if one is open and closes it or not accordingly.
        outputs the info abt position
        :return: True/False for if the check was successful or not.
        """
        if self.position != None:
            print(f"{dt.today()} - Current P/L: {round(self.position.getProfit(),2) if self.position.getProfit() is not None else None} | Peak P/L: {round(self.position.peak,2) if self.position.peak is not None else None} | Margin: {round(self.position.margin,2) if self.position.margin is not None else None}")
            closed, msg = self.position.check()
            print(msg)
            if closed: # i.e if the position actually was closed
                self.pos_manager.record(self.position)
                self.position = None
            if closed == False: # by default closed is None, if it is false that means that the position tried to close
                # but failed.
                print("FAILED TO CLOSE POSITION")
                print("Exiting Trader.")
                exit(0)
        return True
    def finish(self):
        print("Closing position and finishing up ...")
        if self.position:
            closed = self.position.close()
            if closed:
                self.pos_manager.record(self.position)
                self.position = None
            else:
                print("ERROR. POSITION FAILED TO CLOSE. ATTEND TO THIS URGENTLY")
    def whatStep(self):
        """
        tells the caller what step to take next
        :return: string either "CHECK TRADE", "CHECK BALANCE", "TRADE", or "WAIT"
        """
        ny = pytz.timezone("America/New_York")
        refreshTime = dt(2021,9,13,10,0,tzinfo=ny).time()
        tradeTime = dt(2021,9,13,10,5,tzinfo=ny).time()
        closeTime = dt(2021,9,13,16,0,tzinfo=ny).time()
        finishTime = dt(2021,9,13,15,45,tzinfo=ny).time()
        tradeable = lambda: tradeTime < dt.now().astimezone(ny).time() < closeTime
        nearlyClosed = lambda: finishTime <= dt.now().astimezone(ny).time() <= closeTime
        closed = lambda: dt.now().astimezone(ny).time() > closeTime
        refresh = lambda: self.newDay and refreshTime < dt.now().astimezone(ny).time() <= tradeTime

        if closed(): self.newDay = True
        if self.beginning and tradeable():
            self.beginning = False
            return "TRADE"

        if refresh():
            self.newDay = False
            return "REFRESH"
        # time variables to control execution
        hour = dt.today().hour
        min = dt.today().minute
        sec = dt.today().second

        # default the program should wait
        msg = "WAIT"


        # pairs for each process:
        # if (time condition) then do
        # if not time condition then reset go-variable for given step
        if min % 5 == 0 and min != 0 and not (hour == 8 and min == 45 and self.go_for_finish) and self.go_for_bal: # every 5 minutes check balance apart from when placing trade
            msg = "CHECK BALANCE"
            self.go_for_bal = False
        elif min % 5 != 0 : self.go_for_bal = True

        if min == 0 and self.go_for_trade and tradeable():
            msg = "TRADE"
            self.go_for_trade = False
        elif min != 0: self.go_for_trade = True

        if sec == 0 and min != 0 and self.go_for_check and min % 5 != 0 and not (hour == 8 and min == 45 and self.go_for_finish): # every minute apart from trades, balance check, and finish-up
            msg = "CHECK TRADE"
            self.go_for_check = False
        elif sec != 0: self.go_for_check = True

        if nearlyClosed() and self.go_for_finish:
            msg = "FINISH UP"
            self.go_for_finish = False
        elif not nearlyClosed() and not self.go_for_finish:
            self.go_for_finish = True



        return msg

    def welcomeMsg(self):
        print("*--------------------------------------------------------------*")
        print("*----------------------- VERY IMPORTANT -----------------------*")
        print("*--------------------------------------------------------------*")

        print(f"*                          {dt.today()}          *")
        print("*                        STARTING TRADER                       *")
        print("*--------------------------------------------------------------*")

        print(f"Initial Balance: {self.init_bal}")

    def getStock(self,auto=False):
        """
        gets a list of stocks and then asks user which to trade.
        this is rudimentary, in future a classifier will be run and the highest performing stock will be returned.
        :return: string ticker for the stock.
        """
        # gets list of potential stocks using screener
        stocks = getMovers()
        stock_names = [stock.ticker for stock in stocks]

        # pick is the eventual choice
        pick = None
        print("Total list of stocks:")
        print(', '.join(stock_names))
        # goes through list of stocks until one is picked via Y/N questionnaire
        if not auto:
            for stock in stocks:
                print(f"{stock.ticker}: ")
                print(f"\t-Price: {stock.price}")
                print(f"\t-Change: {stock.change}%")
                print(f"\t-Volume: {stock.volume}")
                yn = input("Y/N: ")
                if yn.lower() == 'y':
                    pick = stock.ticker
                    break
        else:
            pick = stocks[4].ticker # in my experience the first stock is never good but the few after are.
            print(f"{pick} has been chosen.")
        if pick == None: # if no stock is picked in the end application exits.
            print("NO STOCK SELECTED")
            print("EXITING PROGRAM")
            exit(0)
        return pick

if __name__ == "__main__":
    trader = Trader()