import pandas as pd
from position import OldPosition
from position import posFromSeries
import os
from datetime import date as d, datetime as dt
from position import Position
class PositionManager:
    """
    this class will manage all the traders positions and keep a record of all the positions.
    the records will be divided, first by instrument and then by day in a folder system:
    e.g records ->
            EURUSD ->
                2021-01-07
                2021-01-08
                etc...
    """
    def __init__(self, symbol):
        self.symbol = symbol
        #get the path for the ticker's directory in the records and the path for today's file within it.
        self.folderpath = os.path.join("records", self.symbol)


        #if the ticker does not have its own directory then create one
        if not os.path.isdir(self.folderpath):
            os.mkdir(self.folderpath)

        self.getRecordFile()
    def getRecordFile(self):
        """
        this function creates a new record with the current date and the symbol given in the constructor
        :return: None
        """
        filepath = os.path.join(self.folderpath, str(d.today()))
        filepath+=".csv"
        if not os.path.isfile(filepath):
            #open and fill the file with column headers as a CSV
            self.records = open(filepath, "w")

            self.records.write("timestamp,position,shares,open price,close price,P/L,status\n")

            self.records.flush()

            #generate pandas DataFrame from the csv file for easier interaction with other python classes.
            self.record_dset = pd.read_csv(filepath)

        else: # if the records file already exists

            #open the file in append mode
            self.records = open(filepath, "a")

            #create pandas DF to pull and add data easily.
            self.record_dset = pd.read_csv(filepath)
    def isCurrentFile(self):
        """
        checks if file being used is current compared to the data.
        i.e file being used is entitled "2021-03-19.csv" (date at time of writing)
        but whilst running application on Forex pair it passes midnight, we don't want the program to continue recording to 2021-03-19
        it needs to record to 2021-03-20.csv
        so this function tells the caller if it is using the current file or not.
        :return: Bool
        """
        hypo_filename = os.path.join(self.folderpath, (str(d.today()) + ".csv"))
        if self.records.name != hypo_filename:
            # if the dates are mismatched
            return False
        else: return True

    def record(self, position:Position):
        """
        this function records a position into the
        :param position:
        :return:
        """
        if not self.isCurrentFile(): #if the file being used is not current:
            self.getRecordFile() #get a new record file to save data to
        #gets variables from the position passed in arguments
        direction = position.direction
        closePrice = position.close_price
        openPrice = position.open_price
        closeTime = position.close_time
        shares = position.shares
        PL = position.pl
        if not any([direction, closePrice, openPrice, closeTime, shares, PL]):
            return False

        #get the end status of the position
        if PL > 0:status = "PROFIT"
        elif PL == 0: status = "NEUTRAL"
        elif position.disallowed: status="DISALLOWED"
        else: status = "LOSS"

        #append a row to the dataframe of positions
        self.record_dset.loc[len(self.record_dset)-1] = pd.Series(index=self.record_dset.columns, data=[closeTime, direction,shares, openPrice, closePrice, PL, status])

        #append a line to the records file for the position and save the file in case the program shuts down for whatever reason
        self.records.write("%s,%s,%s,%s,%s,%s,%s\n" % (str(closeTime),direction,shares, openPrice, closePrice, PL, status))
        self.records.flush()

    # get last x positions, latest position by default
    def getLastPosition(self):
        if len(self.record_dset) > 1:
            return posFromSeries(self.record_dset.tail(1).squeeze()) #.tail() returns a dataframe so use .squeeze() to convert single row df into series
        else:
            return None

    def getLastXPositions(self, x=1):
        positions = []
        for index, row in self.record_dset.tail(x).iterrows():
            positions.append(posFromSeries(row))
