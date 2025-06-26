import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import curve_fit

def linear_regression(X, y):
    model = LinearRegression().fit(X, y)
    y_pred = model.predict(X)
    return model, y_pred

def polynomial_regression(X, y, degree):
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    model = LinearRegression().fit(X_poly, y)
    y_pred = model.predict(X_poly)
    return model, y_pred, poly

def ridge_regression(X, y, alpha):
    model = Ridge(alpha=alpha).fit(X, y)
    y_pred = model.predict(X)
    return model, y_pred

def lasso_regression(X, y, alpha):
    model = Lasso(alpha=alpha).fit(X, y)
    y_pred = model.predict(X)
    return model, y_pred

def elasticnet_regression(X, y, alpha, l1_ratio):
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio).fit(X, y)
    y_pred = model.predict(X)
    return model, y_pred

def nonlinear_exponential(X, y):
    def exp_func(x, a, b):
        return a * np.exp(b * x)
    popt, _ = curve_fit(exp_func, X.flatten(), y)
    a, b = popt
    y_pred = exp_func(X.flatten(), a, b)
    return y_pred, a, b

