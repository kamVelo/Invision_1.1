from selenium.webdriver import Chrome
from selenium.webdriver.chrome.options import Options
from selenium.common.exceptions import JavascriptException,NoSuchElementException, StaleElementReferenceException
from time import sleep
from datetime import datetime as dt
from stock import Stock
import os

def isTradeable_212(stock:Stock):
    """
    this function checks if a stock is available to trade on trading 212.
    :param stock: Stock object
    :return: True/False for if the stock is tradeable or not.
    """
    symbol = stock.ticker
    if dt.now().date().day == 1 or not os.path.isfile("T212Stocks.txt"):
        getT212Stocks()
    record = open("T212Stocks.txt", "r")
    tickers = record.read().split('\n')
    if symbol in tickers:
        return True
    else:
        return False
def getT212Stocks():
    opts = Options()  # options class for selenium
    opts.headless = False  # allows me to see the browser
    browser = Chrome(options=opts)  # initiates browser
    browser.get("https://demo.trading212.com/")  # gets the website

    # gets username and password entry boxes and fills with the username and password.

    """this while loop accounts for the loading time regardless of internet speeds"""
    sleep(3)
    loggedIn = False
    from keys import Keys
    username = Keys.get("t212Username")
    password = Keys.get("t212Password")
    while not loggedIn:
        try:
            username = browser.find_element_by_id("username-real")
            username.send_keys(username)
            password = browser.find_element_by_id("pass-real")
            password.send_keys(password)
            password.submit()
            loggedIn = True
        except JavascriptException:
            print("ERROR")
            pass
    sleep(5)
    browser.find_element_by_id("navigation-search-button").click() #opens search box
    sleep(5)
    browser.find_element_by_css_selector("div[data-folder-key='stocks']").click() #opens up stock folder
    sleep(5)
    # below gets the USA stock folder and clicks on it.
    # ignore childNodes
    browser.execute_script('document.querySelector(\'[data-folder-key="usa"]\').childNodes[0].childNodes[1].click()')
    sleep(5)
    raw_list = browser.find_elements_by_class_name('search-results-instrument')
    record = open("T212Stocks.txt", "w")
    for el in raw_list:
        data = el.text.split('\n')
        ticker = data[0][data[0].find('(')+1:data[0].find(')')]
        record.write(f"{ticker}\n")
    record.close()
def getMovers():
    """
    this function gets all the biggest upward movers from finviz.com under $5
    :return: list of Stock objects
    """

    # starts browser
    url = "https://finviz.com/screener.ashx?v=111&f=sh_price_20to50,ta_gap_u5&ft=3"
    opts = Options()
    opts.headless = False
    browser = Chrome(options=opts)
    browser.get(url)
    browser.maximize_window()
    browser.execute_script("window.scrollTo(0,200);")
    script = """ 
    table = document.getElementsByClassName('table-light')[0]
    let arr = Array.from(table.getElementsByTagName('tr'))
    return arr;
    """
    raw_stocks = browser.execute_script(script)[1:]
    stocks = []
    for raw_stock in raw_stocks:
        try:
            ticker = raw_stock.find_elements_by_tag_name("td")[1].text
            price = float(raw_stock.find_elements_by_tag_name("td")[8].text)
            change = float(raw_stock.find_elements_by_tag_name("td")[9].text[:-1])
            volume = float(raw_stock.find_elements_by_tag_name("td")[10].text.replace(",",""))
            stock = Stock(ticker,price)
            stock.change = change
            stock.volume = volume
            stocks.append(stock)
        except ValueError:
            print(raw_stock.text)
    changes = [abs(stk.change) for stk in stocks]
    avgChange = sum(changes)/len(changes)
    stocks = [stk for stk in stocks if abs(stk.change) < avgChange]
    browser.close()
    return stocks


def getT212Primary():
    """
    this function gets the best stock within the parameters set on finviz.com that is tradeable with T212
    :return: Stock object
    """
    movers = getMovers()
    for opt in movers:
        if isTradeable_212(opt):
            return opt

if __name__ == "__main__":
    instruments = getMovers()
    for instrument in instruments:
        print(instrument.ticker)