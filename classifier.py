# imports
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
import joblib


class Classifier:

    # this evaluates what the signals effectively mean
    def getDirs(self, x, future_closes):
        profit = 0
        # get the profit assuming that 1.0 = buy
        for index, row in x.iterrows():
            prediction = row["classification"]
            close = float(row["60 close"])
            f_close = float(future_closes[index])
            if prediction == 1.0:
                dif = f_close - close
                profit += dif
            elif prediction == 0.0:
                dif = close - f_close
                profit += dif
        profit *= 2000

        # assesses whether the hypothesis above is true and returns a signal list accordingly
        if profit > 0:
            print("PROFIT: %s" % profit)
            return [1.0, 0.0]  # first value is the buy signal, second value is the sell signal ALWAYS
        elif profit < 0:
            print("PROFIT: %s" % profit * -1)
            return [0.0, 1.0]

    def __init__(self, symbol):
        self.symbol = symbol
        # get data remove 14 rows off the end of dset to account for rsi delay, reverse the array so it is in proper CHRONOLOGICAL order
        self.dset = pd.read_csv(os.path.join("data", self.symbol, "5min.csv")).iloc[:-16, 1:].iloc[::-1]
        self.rsi = pd.read_csv(os.path.join("data", self.symbol, "RSI.csv")).iloc[:, 1:].iloc[::-1]
        self.scalerPath = os.path.join("data", symbol, "scaler.bin")
        # add rsi column to the main dset
        self.dset["RSI"] = self.rsi["RSI"]
        print(len(self.dset))
        # get number of hour sections
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
                rsi_name = "%s rsi" % counter
                close_name = "%s close" % counter
                section_series[rsi_name] = row["RSI"]
                section_series[close_name] = row["close"]
                counter += 5
                # if there is a whole hour of data add it to the main dset
                if str(section_series["60 close"]) != "nan":
                    sections = sections.append(section_series, ignore_index=True)
        # resets the main dset to the dset just created above
        self.dset = sections

        # number of categories that the KMeans model can classify into
        num_cats = 2
        
        # Standardise the data
        # if there is a scaler already saved for this stock use that
        if os.path.exists(self.scalerPath):
            scaler = joblib.load(self.scalerPath)
        else:
            scaler = StandardScaler()
        self.scaled = scaler.fit_transform(self.dset)
        joblib.dump(scaler, self.scalerPath, compress=True)

        # fitting the model
        self.model = KMeans(init="random", n_clusters=num_cats, n_init=3, max_iter=300, random_state=42)

        self.model.fit(self.scaled)

        #check for NaN values
        # print(self.dset.isnull().any())

        # add the classification to the dset
        self.dset["classification"] = self.model.labels_
        # adds an index so that we can display hourly data in the show function
        self.dset["index"] = self.dset.index
        # this creates a future closes column so we can figure out what an outputted prediction from the nn means in the function at the top of the class
        closes = self.dset["60 close"].values[1:]
        self.dset = self.dset.iloc[:-1]
        self.dset["future close"] = closes

    # shows chart of the data with a different colour for predictions - not necessary just a nice feature
    def show(self):
        sns.lmplot("index", "60 close", data=self.dset, hue="classification")
        plt.show()


    # prepares the neural network
    def prepNN(self):
        # gets the features from the main dataset
        features = self.dset.iloc[:, :-3].values  # gets rid of future closes, index, and predictions

        # gets the classifications
        labels = self.dset["classification"].values

        # splits the dataset into training and testing feature/label datasets
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

        # getting the correct signals:

        (self.buy_signal, self.sell_signal) = self.getDirs(self.dset, self.dset["future close"])
        print(f"Buy Signal: {self.buy_signal}")
        print(f"Sell Signal: {self.sell_signal}")

    # predicts for a single feature matrix - will throw exception if any of the functions above ( apart from show) are not already called on the same instance.
    def predict(self, feature):
        # uses nn to make prediction vector (i.e not single round number
        pred = self.nn_model.predict(feature)
        # pred = self.model.predict(feature)
        # get the prediction as a single round scalar
        prediction = np.argmax(pred, axis=1)

        # returns signal
        if abs(self.buy_signal - pred) <= 0.2:  # i.e long
            return "BUY"
        elif abs(self.sell_signal - pred) <= 0.2:  # i.e short
            return "SELL"
        else:
            return None


"""
if __name__ == "__main__":
    c = Classifier("ACI")
    c.prepNN()
    c.show()
    features = [list(x) for x in list(c.dset.iloc[:,:-3].values)]
    price = 0
    balance = 500

    for index, row in c.dset.iterrows():
        profit = 0
        feature = np.array(features[index]).reshape(1,-1)
        pred = c.predict(feature)
        diff = row["future close"] - row["60 close"]
        price = row["future close"]
        print(f"prediction: {pred}")
        shares = balance/price
        print(f"Iteration: {index}")
        print(f"{shares=}")
        print(f"{diff=}")
        if pred == 'BUY': #i.e long
            profit += shares * diff
            print("Profit: ", profit)
            balance += profit
        elif pred == 'SELL': #i.e short
            profit += shares * -1 * diff
            print("Profit: ", profit)
            balance += profit
        elif pred == None:
            print("No Decision")
        print(f"Current balance: {balance}\n")
        if balance < 0:
            print("-ve balance exiting program.")
            break

    print(f"Price: {price}")
    print(f"return: {balance - 500}")

"""
if __name__ == '__main__':
    c = Classifier("AAPL")
    c.show()
    #c.prepNN()