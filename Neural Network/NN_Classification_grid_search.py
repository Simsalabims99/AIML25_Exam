import os
import pandas as pd
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from helpMethods.data import to_dataloader, train_val_split
from helpMethods.training import fit
from helpMethods.utils import get_device
from itertools import product

DEVICE = get_device()

df = pd.read_csv('Data/cleaned_data.csv')

def data_preparation(dataframe):
    dataframe = pd.get_dummies(dataframe, columns=['Medlemstype'], drop_first=True)
    dataframe = pd.get_dummies(dataframe, columns=['Region'], drop_first=True)
    dataframe['Aktiv_Deltager'] = dataframe['Aktiv_Deltager'].map({'Ja': 1, 'Nej': 0})

    # Balancing the dataset
    df_majority = dataframe[dataframe['Aktiv_Deltager'] == 0]
    df_minority = dataframe[dataframe['Aktiv_Deltager'] == 1]
    df_minority_upsampled = df_minority.sample(len(df_majority), replace=True, random_state=42)
    dataframe = pd.concat([df_majority, df_minority_upsampled])
    dataframe = dataframe.sample(frac=1, random_state=42)  # Shuffle the dataset

    # Splitting the data into features and target variable
    features = dataframe.drop(columns=['Aktiv_Deltager', 'Kontakt_ID', 'Kontakt_OK', 'Antal_Aktiv', 'Status_Aarsag', 'Status', 'Startdato', 'Medlem_Status', 'Antal'])
    features = features.astype({col: int for col in features.select_dtypes(bool).columns})
    target = dataframe['Aktiv_Deltager']

    return features, target


class TabularDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class NeuralNet(nn.Module):
    def __init__(self, input_dim, hidden1=64, hidden2=32, dropout1=0.3, dropout2=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden1)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout2)
        self.fc3 = nn.Linear(hidden2, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

def plot_loss(history: dict):
   
    plt.figure(figsize=(8, 5))
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.savefig("Visualisations/Loss_curve.png", dpi=300, bbox_inches='tight')

def plot_accuracy(history: dict):

    plt.figure(figsize=(8, 5))
    plt.plot(history["train_acc"], label="Train Accuracy")
    plt.plot(history["val_acc"], label="Validation Accuracy")
    plt.title("Accuracy Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)

    plt.savefig("Visualisations/Accuracy_curve.png", dpi=300, bbox_inches='tight')


def grid_search_nn(param_grid, dateframe):
    results = []

    features, target = data_preparation(dateframe)

    for h1, h2, d, epochs in product(
        param_grid['hidden1'],
        param_grid['hidden2'],
        param_grid['dropout'],
        param_grid['epochs']
    ):
        print(f"Training model with h1={h1}, h2={h2}, d={d}, epochs={epochs}")
        model = NeuralNet(features.shape[1], h1, h2, d, d).to(DEVICE)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        loss_fn = nn.CrossEntropyLoss()

        X_tensor = torch.FloatTensor(features.values)
        y_tensor = torch.LongTensor(target.values)
        X_train_val, X_test, y_train_val, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, stratify=y_tensor, random_state=42)

        train_set = TabularDataset(X_train_val, y_train_val)
        test_set = TabularDataset(X_test, y_test)
        train_set, val_set = train_val_split(train_set, val_ratio=0.2, seed=42)

        train_loader = to_dataloader(train_set, batch_size=64, shuffle=True)
        val_loader = to_dataloader(val_set, batch_size=64, shuffle=False)
        test_loader = to_dataloader(test_set, batch_size=64, shuffle=False)

        model, _ = fit(model, train_loader, val_loader, DEVICE, optimizer, loss_fn ,epochs)

        y_true, y_pred, y_prob = [], [], []
        model.eval()
        with torch.no_grad():
            for feature_batch, label_batch in test_loader:
                feature_batch, label_batch = feature_batch.to(DEVICE), label_batch.to(DEVICE)
                out = model(feature_batch)
                _, predictions = torch.max(out, 1)
                probabilities = torch.softmax(out, dim=1)[:, 1]
                y_true.extend(label_batch.cpu().numpy())
                y_pred.extend(predictions.cpu().numpy())
                y_prob.extend(probabilities.cpu().numpy())

        report = classification_report(y_true, y_pred, output_dict=True)
        auc = roc_auc_score(y_true, y_prob)
        results.append({
            'hidden1': h1, 'hidden2': h2, 'dropout1': d, 'epochs': epochs,
            'accuracy': report['accuracy'],
            'precision': report['1']['precision'],
            'recall': report['1']['recall'],
            'f1': report['1']['f1-score'],
            'auc': auc
        })

    return pd.DataFrame(results) 

def train_nn_model(data):
    # Data preparation
    features, target = data_preparation(data)

    # Splitting the data into training, validation, and test sets
    X_tensor = torch.FloatTensor(features.values)
    y_tensor = torch.LongTensor(target.values)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X_tensor, y_tensor, test_size=0.2, stratify=y_tensor, random_state=42
    )

    train_set = TabularDataset(X_train_val, y_train_val)
    test_set = TabularDataset(X_test, y_test)
    train_set, val_set = train_val_split(train_set, val_ratio=0.2, seed=42)

    train_loader = to_dataloader(train_set, batch_size=64, shuffle=True)
    val_loader = to_dataloader(val_set, batch_size=64, shuffle=False)
    test_loader = to_dataloader(test_set, batch_size=64, shuffle=False)

    model = NeuralNet(X_tensor.shape[1]).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    model, history = fit(model, train_loader, val_loader, DEVICE, optimizer, loss_fn, 20)

    y_true, y_pred, y_prob = [], [], []
    model.eval()
    with torch.no_grad():
        for features_batch, labels_batch in test_loader:
            features_batch, labels_batch = features_batch.to(DEVICE), labels_batch.to(DEVICE)
            out = model(features_batch)
            _, predictions = torch.max(out, 1)
            probabilities = torch.softmax(out, dim=1)  
            y_true.extend(labels_batch.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())
            y_prob.extend(probabilities[:, 1].cpu().numpy())

    plot_loss(history)
    plot_accuracy(history)
    report = classification_report(y_true, y_pred, output_dict=True)
    auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) == 2 else None

    return {
        'accuracy': report['accuracy'],
        'precision': report['1']['precision'],
        'recall': report['1']['recall'],
        'f1': report['1']['f1-score'],
        'auc': auc
    }


# Execution order of the code
def tune_hyperparameters(df):

    param_grid_1 = {
        'hidden1': [64],
        'hidden2': [32],
        'dropout': [0.2, 0.3],
        'epochs': [10,20]
    }

    param_grid_2 = {
        'hidden1': [128],
        'hidden2': [64],
        'dropout': [0.2, 0.3],
        'epochs': [10,20]
    }

    results_df = grid_search_nn(param_grid_1, df)
    results_df_2 = grid_search_nn(param_grid_2, df)
    print(results_df.sort_values(by='f1', ascending=False).head())
    print(results_df_2.sort_values(by='f1', ascending=False).head())


#tune_hyperparameters(df)
print(train_nn_model(df))