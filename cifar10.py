import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE
from umap import UMAP
from sklearn.metrics import accuracy_score
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset
import torch.nn as nn
import torch.optim as optim
from init import Args, device

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes):
        super(NeuralNetwork, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(nn.ReLU())
        for i in range(1, len(hidden_sizes)):
            layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_sizes[-1], num_classes))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)

class MLPClassifier:
    def __init__(self, input_size, hidden_sizes, num_classes):
        self.device = device
        self.model = NeuralNetwork(input_size, hidden_sizes, num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=Args.lr)

    def fit(self, X, y):
        X = torch.FloatTensor(X).to(self.device)
        y = torch.LongTensor(y).to(self.device)
        
        dataset = torch.utils.data.TensorDataset(X, y)
        train_loader = torch.utils.data.DataLoader(dataset, batch_size=Args.batch_size, shuffle=True)
        
        self.model.train()
        training_loss = []    
        for epoch in range(1, Args.total_epochs + 1):
            for batch_idx, (data, target) in enumerate(train_loader):
                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

                if batch_idx % 100 == 0:
                    training_loss.append(loss.item())
                    batch = batch_idx * len(data)
                    data_count = len(train_loader.dataset)
                    percentage = (100. * batch_idx / len(train_loader))
                    print(f'Epoch {epoch}: [{batch:5d} / {data_count}] ({percentage:.0f} %)' + f'  Loss: {np.mean(training_loss):.6f}')

    def predict(self, X):
        X = torch.FloatTensor(X).to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X)
            _, predicted = torch.max(outputs, 1)
        return predicted.cpu().numpy()

def train_and_evaluate(model, X, y):
    X_train, X_test = X[:50000], X[50000:]
    y_train, y_test = y[:50000], y[50000:]
    
    new_model = model(X.shape[1], (1024, 1024), 10)
    new_model.fit(X_train, y_train)
    y_pred = new_model.predict(X_test)
    accuracies = accuracy_score(y_test, y_pred)
    
    return accuracies


def extract_features(model, data_loader):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for batch, targets in data_loader:
            batch = batch.to(device)
            batch_features = model.model.get_feature(batch)
            features.append(batch_features.cpu().numpy())
            labels.append(targets.numpy())
    return np.concatenate(features), np.concatenate(labels)

def load_data(model):
    normalization_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    train_data = torchvision.datasets.CIFAR10(root='./cifar10', train=True, transform=normalization_transform, download=True)
    test_data = torchvision.datasets.CIFAR10(root='./cifar10', train=False, transform=normalization_transform, download=True)
    
    train_loader = DataLoader(train_data, batch_size=1024, shuffle=False, num_workers=12)
    test_loader = DataLoader(test_data, batch_size=1024, shuffle=False, num_workers=12)

    X_train, y_train = extract_features(model, train_loader)
    X_test, y_test = extract_features(model, test_loader)
    return np.concatenate((X_train, X_test), axis=0), np.concatenate((y_train, y_test), axis=0)

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

if __name__ == '__main__':
    print(f"current device is {device}")

    pretrained = {
        'AlexNet': torch.load('./pretrained/alex.pth').to(device).eval(),
        'ResNet-9': torch.load('./pretrained/res9.pth').to(device).eval(),
        'ResNet-152': torch.load('./pretrained/res152.pth').to(device).eval()
    }
    
    X, y = load_data(pretrained['ResNet-152'])
    print(X.shape, y.shape)

    models = {
        'Neural Network': MLPClassifier
    }

    analysis_methods = {
        'Original': None,
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

