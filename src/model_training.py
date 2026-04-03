# src/model_training.py
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

def train_models(X_train, y_train):
    """Train multiple classification models"""
    models = {
        'LogisticRegression': Pipeline([
            ('classifier', LogisticRegression(max_iter=1000, random_state=42))
        ]),
        'KNN': Pipeline([
            ('classifier', KNeighborsClassifier(n_neighbors=5))
        ]),
        'DecisionTree': Pipeline([
            ('classifier', DecisionTreeClassifier(random_state=42))
        ]),
        'RandomForest': Pipeline([
            ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
        ])
    }
    
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
    
    return trained_models
