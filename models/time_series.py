from statsmodels.tsa.arima.model import ARIMA

def arima_forecast(y, order):
    model = ARIMA(y, order=order).fit()
    y_pred = model.forecast(steps=5)  # تتوقع 5 خطوات للأمام
    return y_pred, model

# 🔥 Main Controller Function
def handle_time_series(y, order=(1,1,1)):
    """
    Handles Time Series forecasting using ARIMA.
    """
    y_pred, model = arima_forecast(y, order)
    return y_pred
