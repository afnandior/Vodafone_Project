import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import curve_fit

# Linear Regression
def linear_regression(X, y):
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    return model, y_pred

# Polynomial Regression
def polynomial_regression(X, y, degree):
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    model = LinearRegression().fit(X_poly, y)
    y_pred = model.predict(X_poly)
    return model, y_pred, poly

# Ridge Regression
def ridge_regression(X, y, alpha):
    model = Ridge(alpha=alpha).fit(X, y)
    y_pred = model.predict(X)
    return model, y_pred

# Lasso Regression
def lasso_regression(X, y, alpha):
    model = Lasso(alpha=alpha).fit(X, y)
    y_pred = model.predict(X)
    return model, y_pred

# ElasticNet Regression
def elasticnet_regression(X, y, alpha, l1_ratio):
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio).fit(X, y)
    y_pred = model.predict(X)
    return model, y_pred

# Nonlinear Exponential Regression
def nonlinear_exponential(X, y):
    def exp_func(x, a, b):
        return a * np.exp(b * x)
    popt, _ = curve_fit(exp_func, X.flatten(), y)
    a, b = popt
    y_pred = exp_func(X.flatten(), a, b)
    return y_pred, a, b

# 🔥 Main Controller Function
def handle_regression_models(model_type, X, y):
    """
    Handles regression model selection and execution.

    Parameters:
        model_type (str): Type of regression model to run.
        X (array): Input features.
        y (array): Target values.

    Returns:
        model: Trained model or None for non-linear exponential.
        y_pred: Predicted values.
        extra (dict): Additional info if needed (e.g., polynomial transformer, exponential params).
    """
    if model_type == "Linear Regression":
        model, y_pred = linear_regression(X, y)
        return model, y_pred, {}

    elif model_type == "Polynomial Regression":
        degree = 2  
        model, y_pred, poly = polynomial_regression(X, y, degree)
        return model, y_pred, {"poly": poly}

    elif model_type == "Ridge Regression":
        alpha = 1.0  # يمكنك جعله ديناميكيًا عبر Streamlit لاحقًا
        model, y_pred = ridge_regression(X, y, alpha)
        return model, y_pred, {}

    elif model_type == "Lasso Regression":
        alpha = 0.1  
        model, y_pred = lasso_regression(X, y, alpha)
        return model, y_pred, {}

    elif model_type == "ElasticNet Regression":
        alpha = 0.1
        l1_ratio = 0.5
        model, y_pred = elasticnet_regression(X, y, alpha, l1_ratio)
        return model, y_pred, {}

    elif model_type == "Nonlinear Regression (Exponential)":
        y_pred, a, b = nonlinear_exponential(X, y)
        return None, y_pred, {"a": a, "b": b}

    else:
        return None, None, {}
