import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_breast_cancer, load_digits
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.base import clone
from umap import UMAP

def load_data(load_func):
    data = load_func()
    return data.data, data.target

def DimensionReduction(X, y, method_name, reducer):
    if reducer is not None:
        X_reduced = reducer.fit_transform(X)
    else:
        X_reduced = X

    if reducer is not None:
        plt.figure(figsize=(8, 6))
        plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='viridis')
        plt.title(f'{method_name} of dataset')
        plt.xlabel(f'{method_name} feature 1')
        plt.ylabel(f'{method_name} feature 2')
        plt.colorbar(label='Target')
        plt.show()

    if isinstance(reducer, PCA):
        print(f"Explained variance ratio: {reducer.explained_variance_ratio_}")

    return X_reduced

def train_and_evaluate(model, X, y):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    accuracies = []
    
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        new_model = clone(model)
        new_model.fit(X_train, y_train)
        y_pred = new_model.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred))
    
    return np.mean(accuracies)

if __name__ == '__main__':
    models = {
        'SVM': SVC(),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=5000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'KNN': KNeighborsClassifier(n_neighbors=5),
        'Neural Network': MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)
    }

    X, y = load_data(load_digits)

    # Dictionary of analysis methods
    analysis_methods = {
        'Original': None,
        'PCA': PCA(n_components=2),
        'KernelPCA (poly)': KernelPCA(n_components=2, kernel='poly'),
        'KernelPCA (rbf)': KernelPCA(n_components=2, kernel='rbf'),
        'TSNE': TSNE(n_components=2, random_state=42),
        'UMAP': UMAP(n_components=2, random_state=42)
    }

    for method_name, reducer in analysis_methods.items():
        print(f"\n{method_name} Analysis")
        X_analyzed = DimensionReduction(X, y, method_name, reducer)
        
        for model_name, model in models.items():
            accuracy = train_and_evaluate(model, X_analyzed, y)
            print(f"Current Model is {model_name}, Average accuracy: {accuracy:.8f}")
            print('_________________________________________________')
