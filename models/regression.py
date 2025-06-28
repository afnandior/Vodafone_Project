import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import curve_fit

def linear_regression(X, y):
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    return model, y_pred, {}

def polynomial_regression(X, y, degree=2):
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    model = LinearRegression().fit(X_poly, y)
    y_pred = model.predict(X_poly)
    return model, y_pred, {"poly": poly}

def ridge_regression(X, y, alpha=1.0):
    model = Ridge(alpha=alpha).fit(X, y)
    y_pred = model.predict(X)
    return model, y_pred, {}

def lasso_regression(X, y, alpha=0.1):
    model = Lasso(alpha=alpha).fit(X, y)
    y_pred = model.predict(X)
    return model, y_pred, {}

def elasticnet_regression(X, y, alpha=0.1, l1_ratio=0.5):
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio).fit(X, y)
    y_pred = model.predict(X)
    return model, y_pred, {}

def nonlinear_exponential(X, y):
    def exp_func(x, a, b):
        return a * np.exp(b * x)
    popt, _ = curve_fit(exp_func, X.flatten(), y)
    a, b = popt
    y_pred = exp_func(X.flatten(), a, b)
    return None, y_pred, {"a": a, "b": b}

# 🔥 Main Controller Function
def handle_regression_models(model_type, X, y):
    """
    Selects and runs the regression model based on user selection.
    """
    if model_type == "Linear Regression":
        return linear_regression(X, y)
    elif model_type == "Polynomial Regression":
        return polynomial_regression(X, y, degree=2)  # يمكن جعله ديناميكيًا في streamlit_app.py
    elif model_type == "Ridge Regression":
        return ridge_regression(X, y, alpha=1.0)
    elif model_type == "Lasso Regression":
        return lasso_regression(X, y, alpha=0.1)
    elif model_type == "ElasticNet Regression":
        return elasticnet_regression(X, y, alpha=0.1, l1_ratio=0.5)
    elif model_type == "Nonlinear Regression (Exponential)":
        return nonlinear_exponential(X, y)
    else:
        return None, None, {}
