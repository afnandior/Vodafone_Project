from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

def logistic_regression(X, y):
    model = LogisticRegression().fit(X, y)
    y_pred = model.predict(X)
    return model, y_pred, {}

def decision_tree_classifier(X, y):
    model = DecisionTreeClassifier().fit(X, y)
    y_pred = model.predict(X)
    return model, y_pred, {}

# 🔥 Main Controller Function
def handle_classification_models(model_type, X, y):
    """
    Handles classification model selection and execution.
    """
    if model_type == "Logistic Regression":
        return logistic_regression(X, y)
    elif model_type == "Decision Tree Classification":
        return decision_tree_classifier(X, y)
    else:
        return None, None, {}
