import os
import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import math
import scipy.stats as stats
import matplotlib.pyplot as plt
class Environment:
    """
    this is the reinforcement learning environment
    """
    def __init__(self, ticker:str,test=False):
        ticker = ticker.upper()

        if test: # if this is a test run, use the shorter dataset to expedite the testing
            self.dset = pd.read_csv(ticker + "test.csv").iloc[::-1].iloc[16:,:] # account for RSI being shorter
            self.RSI = pd.read_csv(ticker + "RSI_test.csv").iloc[::-1].iloc[:,1:]
        else: # find full size dataset
            dataPath = os.path.join("data",ticker)
            self.dset = pd.read_csv(os.path.join(dataPath,"5min.csv")).iloc[::-1].iloc[16:,:]
            self.RSI = pd.read_csv(os.path.join(dataPath, "RSI.csv")).iloc[::-1].iloc[:, 1:]

        # reset index so it's not reversed
        self.dset.index = range(len(self.dset))

        # attach RSI values to dataset
        self.dset["RSI"] = self.RSI["RSI"]

        # for each row calculate values below:
        times = []
        rewards = []
        for index,row in self.dset.iterrows():

            # convert datetime to minutes past midnight
            timeStr = row["time"].split(" ")[1]
            hours = float(timeStr.split(":")[0])
            mins = float(timeStr.split(":")[1])
            mins += hours * 60
            times.append(mins)


        self.dset["time"] = times

        num_sects = round(len(self.dset) / 12)

        columns = []  # column names
        # different times between 5-60 inclusive that data is recorded from
        times = list(range(5, 65, 5))

        # create column names by appending close and rsi for each one
        for t in times:
            open_name = "%s open" % t
            high_name = "%s high" % t
            low_name = "%s low" % t
            close_name = "%s close" % t
            vol_name = "%s volume" % t
            rsi_name = "%s rsi" % t
            #columns.extend([open_name,high_name,low_name,close_name,vol_name,rsi_name])
            columns.extend([close_name,rsi_name])
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
                #open_name = "%s open" % counter
                #high_name = "%s high" % counter
                #low_name = "%s low" % counter
                close_name = "%s close" % counter
                #vol_name = "%s volume" % counter
                rsi_name = "%s rsi" % counter
                #section_series[open_name] = row["open"]
                #section_series[high_name] = row["high"]
                #section_series[low_name] = row["low"]
                section_series[close_name] = row["close"]
                #section_series[vol_name] = row["volume"]
                section_series[rsi_name] = row["RSI"]

                counter += 5
                # if there is a whole hour of data add it to the main dset
                if str(section_series["60 close"]) != "nan":
                    sections = sections.append(section_series, ignore_index=True)
        # resets the main dset to the dset just created above
        self.dset = sections
        rewards = self.calcRewards()

        # fill out columns in dataset
        buyRewards = []
        passRewards = []
        sellRewards = []
        decisions = []
        for arr in rewards:
            print(arr)
            buyRewards.append(arr[0])
            passRewards.append(arr[1])
            sellRewards.append(arr[2])
            decision = np.argmax(arr)
            print(decision)
            decisions.append(decision)
        print(decisions)
        self.dset["Buy Reward"] = buyRewards
        self.dset["Pass Reward"] = passRewards
        self.dset["Sell Reward"] = sellRewards
        self.scaler = StandardScaler()
        self.scaled = self.scaler.fit_transform(self.dset.iloc[:,:-3])

        self.scaled = pd.DataFrame(self.scaled,index=self.dset.index, columns=self.dset.columns[:-3])
        self.scaled["decision"] = decisions
        #self.showNormDist("Buy Reward")
        #self.showNormDist("Sell Reward")
        self.prepNN()

    def showNormDist(self,colName:str):
        """
        this plots the distribution of a given column
        :param colName: string column
        :return: None
        """
        x = self.dset[colName]
        mu = sum(x)/len(x)
        variance = (sum([x_**2 for x_ in x]))/len(x) - mu**2
        sigma = math.sqrt(abs(variance))
        print(colName, " - spread and centrality information")
        print("Mean: ", round(mu,2))
        print("Standard Deviation: ", round(sigma,2))

        plt.title(f"{colName} - Normally Distributed")
        plt.scatter(x, stats.norm.pdf(x,mu,sigma))
        for i in range(1,4): # plot three S.D lines:
            plt.axvline(x = mu + (i * sigma), color="r")
            plt.axvline(x = mu - (i * sigma), color="r")
        plt.show()
    def prepNN(self) -> None:
        """
        creates and trains neural network
        :return: None
        """
        features = self.scaled.iloc[:,:-1].values
        labels = self.scaled.iloc[:,-1:].values

        x_train, self.x_test, y_train, self.y_test = train_test_split(features, labels, test_size=0.2)

        # creating the neural network
        self.nn_model = Sequential()
        self.nn_model.add(Dense(64, activation="tanh"))
        self.nn_model.add(Dense(32, activation="tanh"))
        self.nn_model.add(Dense(16, activation="tanh"))
        self.nn_model.add(Dense(1, activation="sigmoid"))

        # fitting the neural network
        self.nn_model.compile(loss="binary_crossentropy", metrics=["accuracy"], optimizer="adam")
        self.nn_model.fit(x_train, y_train, batch_size=32, epochs=100)

        # tests the neural network on the test set and prints the accuracy and loss
        scores = self.nn_model.evaluate(self.x_test, self.y_test, verbose=0)
        print('Accuracy on test data: {}% \n Error on test data: {}%'.format(round(scores[1] * 100, 2),
                                                                             round((1 - scores[1]) * 100, 2)))

        profit = self.testNN()
        print("Profit after going through whole dataset is: ", profit, "$")


    def calcRewards(self) -> list:
        """
        this function calculates the rewards for possible actions in situations.
        :return: list of rewards
        """
        rewards = [] # list of action-reward values, in this order: 0 - buy, 1 - sell, 2 - do nothing
        for index,row in self.dset.iterrows():
            initClose = float(row["5 close"])
            cumulDiffs = 0
            weightedDiffs = 0
            for i in range(index+1,index+8):
                try:
                    close_i = float(self.dset.iloc[i,:]["5 close"])
                    currentDiff = close_i - initClose - cumulDiffs # account for previous rises/falls i.e if u have 3 periods $2.20, $2.40, $2.90 - the rise in the third period is 50 cents not 70 cents
                    cumulDiffs += close_i - initClose
                    #weightedDiffs += (i-index)/2 * currentDiff # weighted to prioritise later profits
                    weightedDiffs += currentDiff  # no weighting - test case
                except IndexError: # this occurs if there are no more rows.
                    break
            rewards.append([weightedDiffs - 3, -0.1 * abs(weightedDiffs), -1 * weightedDiffs - 3]) # - 3 to account for commission; - 0.1 to account for lost profit
        return rewards

    def predict(self,feature):
        """
        this function makes encapsulated predictions
        :return: string prediction
        """
        options = ["BUY", None, "SELL"]
        print(feature)
        feature = self.scaler.transform(np.array(feature[0]).reshape(1,-1))
        print(feature)
        #print(self.scaled)
        rawPred = self.nn_model.predict(feature)
        print(rawPred)
        pred = 2 - int(rawPred[0][0])
        print(pred)
        pred = int(2 - round(rawPred[0][0], 0))
        print(pred)
        return options[pred]

    def testNN(self) -> int:
        """
        this function tests the neural networks approximation
        :return: int profit
        """
        totalProfit = 0
        balance = 100000
        currentPos = 1 # i.e None
        totalProfitOverTime = []
        profits = []
        positions = []
        for index,row in self.dset.iterrows():
            if balance < 0:
                print("Balance is negative fuckwit")
                break
            currClose = float(row["5 close"])
            prevClose = float(self.dset.iloc[index-1,:]["5 close"])
            posSize = int(balance / prevClose) # number of shares assuming you invest whole portfolio every time
            if currentPos == 0: # i.e long
                profit = (currClose - prevClose) * posSize
            elif currentPos == 2: # i.e short
                profit = (prevClose - currClose) * posSize
            else:
                profit = 0
            profits.append(profit)
            totalProfit += profit
            totalProfitOverTime.append(totalProfit)
            balance += profit
            pred = self.nn_model.predict(self.scaled.iloc[index,:-1].values.reshape(1,-1))
            #print(np.argmax(predRewards[0]))
            currentPos = 2 - round(pred[0][0], 0)
            positions.append(currentPos)
            #currentPos = np.argmax(predRewards[0])
            #  standard deviation above the median reward ( one sd since the rewards are all standard scaled)
        plt.title("profits over run through")
        plt.plot(list(range(len(totalProfitOverTime))), totalProfitOverTime)
        #plt.show()
        print("Biggest win/loss: ", profits[np.argmax([abs(x) for x in profits])])
        print(positions)
        return round(totalProfit,2)









if __name__ == '__main__':
    env = Environment("AAPL", True)