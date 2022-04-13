import random
from string import ascii_letters
import requests as rq
from time import sleep
import os
import pandas as pd
from keys import Keys
def randomString(length=6):
    word = ""
    nums = [str(i) for i in range(9)]
    alphas = nums + list(ascii_letters)
    for i in range(length):
        word += random.choice(alphas)
    return word

def download(ticker:str) -> None:
    """
    this function downloads the last two years of 5 minute intraday data for the stock in question and
    stitches it into one file
    :param ticker: string ticker of the stock
    :return: None
    """
    if not os.path.isdir(os.path.join("data", ticker)):
        os.mkdir(os.path.join("data", ticker))
    path = os.path.join("data",ticker,"5min.csv")
    f = open(path,"wb")
    f.write(b"time,open,high,low,close,volume\n")
    count = 0
    for y in range(1,3):
        for m in range(1,12):
            if count % 5 == 0:
                sleep(75)
            url = "https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY_EXTENDED&symbol=%s&interval=5min&slice=year%smonth%s&adjusted=false&apikey=%s" % (ticker, y,m,randomString())
            content = rq.get(url).content
            split = content.decode().split("\n")
            split = split[1:]
            
            f.write(bytes("\n".join(split),encoding="utf-8"))
            f.flush()
            count += 1
    f.close()
    df = pd.read_csv(path).iloc[::-1]
    RSIs = calcRSIArray(df)
    f_RSI = open(os.path.join("data", ticker, "RSI.csv"), "w")
    f_RSI.write("time,RSI\n")
    for row in RSIs:
        f_RSI.write(",".join(row)+"\n")
    print("downloaded")
def calcRSIArray(data:pd.DataFrame,period=14) -> list:
    """
    this function calculates a list of RSI values for a whole dataset as opposed to just 14 days
    :param data: whole pandas dataframe
    :param period: e.g 14 periods
    :return: list of RSI values
    """
    RSIs = []
    for i in range(len(data)-period-1):
        timestamp = data.iloc[i+15]["time"]
        RSI = calcRSI(data.iloc[i:i+15],period=period)
        RSIs.append([str(timestamp),str(RSI)])
    RSIs.reverse()
    return RSIs
def calcRSI(data:pd.DataFrame, period=14) -> int:
    """
    this function calculates the Relative Strength Index from n+1 rows of data
    :param data: dataframe containing data to be calculated
    :return: int rsi
    """
    gains = 0
    losses = 0
    numGains = 0
    numLosses = 0
    for i in range(1,period+1):
        row = data.iloc[i]
        try:
            close = float(row["close"])
            prevClose = float(data.iloc[i - 1]["close"])
        except ValueError:
            break
        diff = close-prevClose
        if diff > 0:
            gains += diff
            numGains += 1
        else:
            losses += diff
            numLosses += 1
    avgGain = gains/14
    avgLoss = abs(losses/14)
    RS = avgGain/avgLoss if avgLoss > 0 else 1
    RSI = 100 - (100/(1+RS))
    return round(RSI,2)
def getFeature(symbol:str):
    fmPrep = Keys.get("fmPrep")
    price_url = "https://financialmodelingprep.com/api/v3/historical-chart/5min/"+symbol+"?apikey=" + fmPrep
    if len(symbol) < 6:# if the symbol is shorter than 6 characters (i.e a stock)
        rsi_url = "https://financialmodelingprep.com/api/v3/technical_indicator/5min/"+symbol+"?period=14&type=rsi&apikey=" + fmPrep
        df = pd.read_json(rq.get(rsi_url).content)[["close", "rsi"]].iloc[0:12,:].iloc[::-1]
    elif len(symbol) == 6:
        rsi_url = "https://www.alphavantage.co/query?function=RSI&symbol=%s&interval=5min&time_period=14&series_type=close&apikey=%s&datatype=csv&outputsize=full" % (symbol, randomString(9))
        price_url = "https://financialmodelingprep.com/api/v3/historical-chart/5min/" + symbol + "?apikey=" + fmPrep
        df = pd.DataFrame(data=pd.read_json(rq.get(price_url).content)["close"], columns=["close"])
        rsi_df = pd.read_csv(io.StringIO(rq.get(rsi_url).content.decode('utf-8')))["RSI"]
        df["rsi"] = rsi_df
        df = df.iloc[0:12, :].iloc[::-1]
    feature = [[]]
    for item in df.values:
        feature[0].extend(list(item))
    return feature
if __name__ == '__main__':
    download("MSFT")
    #df = pd.read_csv("AAPLtest.csv").iloc[1:].iloc[::-1]
    #df.index = range(0,len(df))
