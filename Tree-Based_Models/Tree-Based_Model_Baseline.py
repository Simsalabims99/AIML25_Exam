# Importing the needed libraries
import pandas as pd
from sklearn import tree
import joblib
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import xgboost as xgb

df = pd.read_csv('Data/cleaned_data.csv')

# Get dummy variables for categorical features
def data_preparation(dataframe):
    dataframe = pd.get_dummies(dataframe, columns=['Medlemstype'], drop_first=True)
    dataframe = pd.get_dummies(dataframe, columns=['Region'], drop_first=True)
    dataframe['Aktiv_Deltager'] = dataframe['Aktiv_Deltager'].map({'Ja': 1, 'Nej': 0})

    # Balancing the dataset
    df_majority = dataframe[dataframe['Aktiv_Deltager'] == 0]
    df_minority = dataframe[dataframe['Aktiv_Deltager'] == 1]
    df_minority_upsampled = df_minority.sample(len(df_majority), replace=True, random_state=42)
    dataframe = pd.concat([df_majority, df_minority_upsampled])
    dataframe = dataframe.sample(frac=1, random_state=42) 

    # Splitting the data into features and target variable
    features = dataframe.drop(columns=['Aktiv_Deltager', 'Kontakt_ID', 'Kontakt_OK', 'Antal_Aktiv', 'Status_Aarsag', 'Status', 'Startdato', 'Medlem_Status', 'Antal'])
    target = dataframe['Aktiv_Deltager']

    # Splitting the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test

def train_decision_tree(X_train, y_train, X_test, y_test):
    # Defining parameters for Decision Tree
    dt_parameters = {
        'criterion': 'gini',
        'max_depth': 5,
        'min_samples_split': 5,
        'min_samples_leaf': 2
    }

    # Training the Decision Tree model
    dt_clf = DecisionTreeClassifier(**dt_parameters, random_state=42)
    dt_clf.fit(X_train, y_train)

    # Making predictions
    y_pred_dt = dt_clf.predict(X_test)

    # Evaluating the model
    accuracy_dt = accuracy_score(y_test, y_pred_dt)
    cross_val_dt = cross_validate(dt_clf, X_train, y_train, cv=5)

    print("Decision Tree Accuracy:", accuracy_dt)
    print(classification_report(y_test, y_pred_dt))
    print("Confusion Matrix:\n", pd.crosstab(y_test, y_pred_dt, rownames=['Actual'], colnames=['Predicted'], margins=True))
    print("Cross-validation scores:", cross_val_dt['test_score'])
    print("Mean cross-validation score:", cross_val_dt['test_score'].mean())

    # Visualizing the Decision Tree
    plt.figure(figsize=(20, 10))
    tree.plot_tree(dt_clf, filled=True)
    plt.savefig('Visualisations/decision_tree_baseline.png')

def train_random_forest(X_train, y_train, X_test, y_test):
    # Defining parameters for Random Forest
    rf_parameters = {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2
    }

    # Training the Random Forest model
    rf_clf = RandomForestClassifier(**rf_parameters, random_state=42)
    rf_clf.fit(X_train, y_train)

    # Making predictions
    y_pred_rf = rf_clf.predict(X_test)

    # Evaluating the model
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    cross_val_rf = cross_validate(rf_clf, X_train, y_train, cv=5)

    print("Random Forest Accuracy:", accuracy_rf)
    print(classification_report(y_test, y_pred_rf))
    print("Confusion Matrix:\n", pd.crosstab(y_test, y_pred_rf, rownames=['Actual'], colnames=['Predicted'], margins=True))
    print("Cross-validation scores:", cross_val_rf['test_score'])
    print("Mean cross-validation score:", cross_val_rf['test_score'].mean())

def train_xgboost(X_train, y_train, X_test, y_test):
    # Defining parameters for XGBoost
    xgb_parameters = {
        'n_estimators': 50,
        'max_depth': 5,
        'learning_rate': 0.1,
        'subsample': 0.8
    }

    # Training the XGBoost model
    xgb_clf = xgb.XGBClassifier(**xgb_parameters, random_state=42)
    xgb_clf.fit(X_train, y_train)

    # Making predictions
    y_pred_xgb = xgb_clf.predict(X_test)

    # Evaluating the model
    accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
    cross_val_xgb = cross_validate(xgb_clf, X_train, y_train, cv=5)

    print("XGBoost Accuracy:", accuracy_xgb)
    print(classification_report(y_test, y_pred_xgb))
    print("Confusion Matrix:\n", pd.crosstab(y_test, y_pred_xgb, rownames=['Actual'], colnames=['Predicted'], margins=True))
    print("Cross-validation scores:", cross_val_xgb['test_score'])
    print("Mean cross-validation score:", cross_val_xgb['test_score'].mean())

# Exuction order of code
def train_models(df):
    X_train, X_test, y_train, y_test = data_preparation(df)
    train_decision_tree(X_train, y_train, X_test, y_test)
    train_random_forest(X_train, y_train, X_test, y_test)
    train_xgboost(X_train, y_train, X_test, y_test)


train_models(df)
    