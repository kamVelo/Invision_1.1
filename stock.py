class Stock:
    def __init__(self, ticker:str, price:float):
        #setting assigned variables
        self._ticker = ticker
        self._price = price

        #declaring null vars
        self._change = None
        self._volume = None

    @property
    def ticker(self) -> str:
        return self._ticker

    @property
    def price(self) -> float:
        return self._price

    @price.setter
    def price(self, price:float):
        try:
            if price < 0:
                return False
            else:
                self._price = price
                return True
        except TypeError:
            print(f"Price value passed was not of type float or int, instead it was: {type(price)}")
            return False

    @property
    def change(self) -> float:
        return self._change

    @change.setter
    def change(self, change:float):
        self._change = change

    @property
    def volume(self) -> int:
        return self._volume

    @volume.setter
    def volume(self,volume:int):
        self._volume = volume
