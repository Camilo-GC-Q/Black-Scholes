import numpy as np
import pandas as pd
import scipy.stats as si
import matplotlib.pyplot as plt
import yfinance as yf
import mplfinance as mpf
import plotly.graph_objects as go
from datetime import datetime

class BlackScholesModel:
    def __init__(self, S, K, T, r, sigma):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        
    def d1(self):
        return (np.log(self.S/self.K) + (self.r + (self.sigma**2)/2) * self.T) / (self.sigma * np.sqrt(self.T))
    
    def d2(self):
        return self.d1() - self.sigma * np.sqrt(self.T)

    def call_option_price(self):
        return (self.S * si.norm.cdf(self.d1(), 0, 1) - self.K * np.exp(-self.r * self.T) * si.norm.cdf(self.d2(), 0, 1))
    
    def put_option_price(self):
        return (self.K * np.exp(-self.r * self.T) * si.norm.cdf(-self.d2(), 0, 1) - self.S * si.norm.cdf(-self.d1(), 0, 1))
    
bsm = BlackScholesModel(S = 100, K = 105, T = 7/365, r = 0.05, sigma = 0.02)
print(f"Call Option Price: {round(bsm.call_option_price(), 2)}")
print(f"Put Option Price: {round(bsm.put_option_price(), 2)}")
    
class BlackScholesGreeks(BlackScholesModel):
    def delta_call(self):
        return si.norm.cdf(self.d1(), 0, 1)
    
    def delta_put(self):
        return -si.norm.cdf(-self.d1(), 0, 1)
    
    def gamma(self):
        return si.norm.pdf(self.d1(), 0, 1) / (self.S * self.sigma * np.sqrt(self.T))
    
    def theta_call(self):
        return (-self.S * si.norm.pdf(self.d1(), 0, 1) * self.sigma / (2 * np.sqrt(self.T))) - (self.r * self.K * np.exp(-self.r * self.T) * si.norm.cdf(self.d2(), 0, 1))
    
    def theta_put(self):
        return (-self.S * si.norm.pdf(self.d1(), 0, 1) * self.sigma / (2 * np.sqrt(self.T))) + (self.r * self.K * np.exp(-self.r * self.T) * si.norm.cdf(-self.d2(), 0, 1))
    
    def vega(self):
        return self.S * np.sqrt(self.T) * si.norm.pdf(self.d1(), 0, 1)
    
    def rho_call(self):
        return self.K * self.T * np.exp(-self.r * self.T) * si.norm.cdf(self.d2(), 0, 1)
    
    def rho_put(self):
        return -self.K * self.T * np.exp(-self.r * self.T) * si.norm.cdf(-self.d2(), 0, 1)
    
bsg = BlackScholesGreeks(S = 100, K = 105, T = 1, r = 0.05, sigma = 0.2)
print(f"Call Delta: {bsg.delta_call()}")
print(f"Put Delta: {bsg.delta_put()}")

stock_prices = np.linspace(0, 200, 100) 
deltas = [BlackScholesGreeks(S = price, K = 100, T = 1, r = 0.05, sigma = 0.2).delta_call() for price in stock_prices]

plt.figure(figsize = (10, 5))
plt.plot(stock_prices, deltas)
plt.title("Delta of a Call Option as Underlying Price Changes")
plt.xlabel("Stock Price")
plt.ylabel("Delta")
plt.grid(True)

def plot_option_sensitivity(bs_model, parameter, values, option_type = "call"):
    prices = []
    for val in values:
        setattr(bs_model, parameter, val)
        if option_type == "call":
            prices.append(bs_model.call_option_price())
        else:
            prices.append(bs_model.put_option_price())

    plt.figure(figsize = (10, 5))
    plt.plot(values, prices)
    plt.title(f"Option Price Sensitivity to {parameter.capitalize()}")
    plt.xlabel(parameter.capitalize())
    plt.ylabel("Option Price")
    plt.grid(True)
    
volatility = np.linspace(0.1, 0.3, 100)
plot_option_sensitivity(bsm, "sigma", volatility, "call")
    

def plot_data(tick_sym):
    ticker = yf.Ticker(tick_sym)
    data = ticker.history(period = "1y")
    
    plt.figure(figsize=(10, 5))
    plt.plot(data.index, data['Close'], label = 'Closing Price', color = 'blue')
    
    plt.title(f"{tick_sym.upper()} Historical Stock Price (1 Year)")
    plt.xlabel("Date")
    plt.ylabel("Stock Price")
    plt.grid(True)
    plt.legend()
    plt.show()
    
plot_data('AAPL')

def historical_volatility(stock_data, window = 252): # 252 trading days in a year
    log_returns = np.log(stock_data['Close'] / stock_data['Close'].shift(1))
    volatility = np.sqrt(window) * log_returns.dropna().std()
    return volatility

ticker = yf.Ticker('AAPL')
stock_data = ticker.history(period = '1y')
aapl_vol = historical_volatility(stock_data)
print(f"Apple Historical Volatility: {aapl_vol}")
    
    
def get_options(tick_symbol):
    ticker = yf.Ticker(tick_symbol)
    option_date = ticker.options
    option_data = ticker.option_chain(option_date[0])
    print(ticker.option_chain(option_date[0]))
    return option_data.calls, option_data.puts

# aapl_calls, aapl_puts = get_options('aapl') # example
# print(aapl_calls, aapl_puts)