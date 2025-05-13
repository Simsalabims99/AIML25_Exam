import os
import copy
import pandas as pd
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
from sklearn.utils import resample

import torch
import torch.nn as nn
import torch.optim.sgd
from torch.utils.data import Dataset

from helpMethods.data import to_dataloader, train_val_split
from helpMethods.training import fit
from helpMethods.utils import get_device

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
    def __init__(self, input_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

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
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    loss_fn = nn.CrossEntropyLoss()

    model, _ = fit(model, train_loader, val_loader, DEVICE, optimizer, loss_fn, 10)

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

    report = classification_report(y_true, y_pred, output_dict=True)
    auc = roc_auc_score(y_true, y_prob) if len(set(y_true)) == 2 else None

    return {
        'accuracy': report['accuracy'],
        'precision': report['1']['precision'],
        'recall': report['1']['recall'],
        'f1': report['1']['f1-score'],
        'auc': auc
    }


print(train_nn_model(df))