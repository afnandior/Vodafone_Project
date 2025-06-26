from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

def logistic_regression(X, y):
    model = LogisticRegression().fit(X, y)
    y_pred = model.predict(X)
    return model, y_pred

def decision_tree_classifier(X, y):
    model = DecisionTreeClassifier().fit(X, y)
    y_pred = model.predict(X)
    return model, y_pred

