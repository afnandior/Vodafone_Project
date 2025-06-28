from statsmodels.tsa.arima.model import ARIMA

def arima_forecast(y, order=(1,1,1), steps=5):
    model = ARIMA(y, order=order)
    model_fit = model.fit()
    y_pred = model_fit.predict(start=0, end=len(y)+steps-1)
    return model_fit, y_pred

# 🔥 Main Controller Function
def handle_time_series_models(y, order=(1,1,1), steps=5):
    """
    Handles Time Series ARIMA model.
    """
    return arima_forecast(y, order, steps)
