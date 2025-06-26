from statsmodels.tsa.arima.model import ARIMA

def arima_forecast(y, order):
    model = ARIMA(y, order=order).fit()
    y_pred = model.predict(start=0, end=len(y)+5)
    return model, y_pred
