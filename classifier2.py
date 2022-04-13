from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

class Classifier:

    def __init__(self,ticker:str,test=False):
        """
        constructor for Classifier
        :param ticker: string ticker of stock
        :param test: bool is this a test or not (if so use shortened test data)
        """
        ticker = ticker.upper()
        if not test:
            dataPath = os.path.join("data",ticker)
            self.dset = pd.read_csv(os.path.join(dataPath,"5min.csv")).iloc[::-1].iloc[16:,:]
            self.RSI = pd.read_csv(os.path.join(dataPath,"RSI.csv")).iloc[::-1].iloc[:,1:]
            self.dset["RSI"] = self.RSI["RSI"]
            self.dset = self.splitIntoDays()
        else:
            self.dset = pd.read_csv("AAPLtest.csv").iloc[::-1].iloc[16:, :]
            self.RSI = pd.read_csv("AAPLRSI_test.csv").iloc[::-1].iloc[:, 1:]
            self.dset["RSI"] = self.RSI["RSI"]
            self.dset = self.splitIntoDays()

        for i in range(len(self.dset)):
            df = self.dset[i]
            scaler = StandardScaler()
            self.dset[i] = pd.DataFrame(scaler.fit_transform(df))

        self.dset = pd.concat(self.dset,ignore_index=True) # merges all the dataframes into one

        num_sects = round(len(self.dset) / 12)

        columns = []  # column names
        # different times between 5-60 inclusive that data is recorded from
        times = list(range(5, 65, 5))

        # create column names by appending close and rsi for each one
        for t in times:
            close_name = "%s close" % t
            rsi_name = "%s rsi" % t
            columns.extend([rsi_name, close_name])

        # create pandas dataframe where each row is a hour-long feature going from 5-60 (inclusive) close and 5-60 rsi
        sections = pd.DataFrame(columns=columns)

        # for every hours worht of data in the main dset
        for i in range(0, num_sects):
            # gets the section
            section = self.dset.iloc[i * 12:i * 12 + 12, :]
            # creates a pandas series which will be added as a row to the pandas dataframe called sections
            section_series = pd.Series(index=sections.columns, dtype="float64")
            # counts the times
            counter = 5
            # goes through the section which we have extracted above and gets each 5 minutes of data then adds it to the series for the current hour
            for index, row in section.iterrows():
                print(row)
                rsi_name = "%s rsi" % counter
                close_name = "%s close" % counter
                section_series[rsi_name] = row[5]
                section_series[close_name] = row[3]
                counter += 5
                # if there is a whole hour of data add it to the main dset
                if str(section_series["60 close"]) != "nan":
                    sections = sections.append(section_series, ignore_index=True)
        # resets the main dset to the dset just created above
        self.dset = sections
        num_cats = 2
        self.model = KMeans(init="random", n_clusters=num_cats, n_init=3, max_iter=300, random_state=42)

        self.model.fit(self.dset)

        # check for NaN values
        # print(self.dset.isnull().any())

        # add the classification to the dset
        self.dset["classification"] = self.model.labels_
        # adds an index so that we can display hourly data in the show function
        self.dset["index"] = self.dset.index
        # this creates a future closes column so we can figure out what an outputted prediction from the nn means in the function at the top of the class
        closes = self.dset["60 close"].values[1:]
        self.dset = self.dset.iloc[:-1]
        self.dset["future close"] = closes

    def show(self):
        sns.lmplot("index", "60 close", data=self.dset, hue="classification")
        plt.show()




    def splitIntoDays(self) -> list:
        """
        this function splits the dataset into daily sections
        :return: list of dataframes
        """
        dfs = []
        currDate = float("inf")

        for index,row in self.dset.iterrows():
            date = row["time"].split(" ")[0]
            if date != currDate:
                currDate = date
                try:
                    dfs.append(df)
                except Exception:
                    pass # this will happen on first iteration
                df = pd.DataFrame(columns = self.dset.columns[1:])
            del row["time"]
            df = df.append(row,ignore_index=True)
        return dfs

if __name__ == '__main__':
    c = Classifier("AAPL",test=True)
    c.show()