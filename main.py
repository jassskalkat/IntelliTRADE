pip install prophet pystan yfinance

# libraries needed for the model
import matplotlib.pyplot as plt
import yfinance as yf
from prophet import Prophet

# function for prediction based of the parameters
def prediction(ticker, date, days):
    df = yf.download(ticker, start=date)
    df = df.reset_index()
    df[['ds', 'y']] = df[['Date', 'Adj Close']]
    df['ds'] = df['ds'].dt.tz_localize(None) #remove timezone information
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(days)
    forecast = model.predict(future)
    model.plot(forecast)
    plt.show()

# invoking the function with the arguments
prediction('TSLA', '2011-01-01', 30)