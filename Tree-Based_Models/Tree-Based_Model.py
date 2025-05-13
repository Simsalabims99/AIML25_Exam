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
    
def hyperparameter_tuning(X_train, y_train):
    # Performing cross-validation and hyperparameter tuning
    rf_parameters = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    dt_parameters = {
        'criterion': ['gini', 'entropy'],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    xgb_parameters = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0]
    }

    rf_clf_grid = RandomForestClassifier(random_state=42)
    grid_search_rf = GridSearchCV(estimator=rf_clf_grid, param_grid=rf_parameters, cv=5, scoring='f1', n_jobs=-1, verbose=2)
    grid_search_rf.fit(X_train, y_train)
    print("Best hyperparameters:", grid_search_rf.best_params_)

    dt_clf_grid = DecisionTreeClassifier(random_state=42)
    grid_search_dt = GridSearchCV(estimator=dt_clf_grid, param_grid=dt_parameters, cv=5,scoring='f1', n_jobs=-1, verbose=2)
    grid_search_dt.fit(X_train, y_train)
    print("Best hyperparameters:", grid_search_dt.best_params_)

    xgb_clf_grid = xgb.XGBClassifier(random_state=42)
    grid_search_xgb = GridSearchCV(estimator=xgb_clf_grid, param_grid=xgb_parameters, cv=5, scoring='f1', n_jobs=-1, verbose=2)
    grid_search_xgb.fit(X_train, y_train)
    print("Best hyperparameters:", grid_search_xgb.best_params_)

    return grid_search_rf.best_params_, grid_search_dt.best_params_, grid_search_xgb.best_params_


def train_decision_tree(X_train, y_train, X_test, y_test, dt_hyperparameters):
    # Training the Decision Tree model
    dt_clf = DecisionTreeClassifier(**dt_hyperparameters, random_state=42)
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
    plt.savefig('Visualisations/decision_tree.png')

    # Feature importance
    feature_importances = dt_clf.feature_importances_
    feature_importances_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances})
    feature_importances_df = feature_importances_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    plt.barh(feature_importances_df['Feature'], feature_importances_df['Importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance')
    plt.savefig('Visualisations/feature_importance_dt.png')

    filename = 'Models/decision_tree_model.pkl'
    joblib.dump(dt_clf, filename)

def train_random_forest(X_train, y_train, X_test, y_test, rf_hyperparameters):
    # Training the Random Forest model
    rf_clf = RandomForestClassifier(**rf_hyperparameters, random_state=42)
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

    # Feature importance
    feature_importances = rf_clf.feature_importances_
    feature_importances_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances})
    feature_importances_df = feature_importances_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    plt.barh(feature_importances_df['Feature'], feature_importances_df['Importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance')
    plt.savefig('Visualisations/feature_importance_rf.png')    

    filename = 'Models/random_forest_model.pkl'
    joblib.dump(rf_clf, filename)

def train_xgboost(X_train, y_train, X_test, y_test, xgb_hyperparameters):
    # Training the XGBoost model
    xgb_clf = xgb.XGBClassifier(**xgb_hyperparameters, random_state=42)
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

    # Feature importance
    feature_importances = xgb_clf.feature_importances_
    feature_importances_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': feature_importances})
    feature_importances_df = feature_importances_df.sort_values(by='Importance', ascending=False)

    plt.figure(figsize=(10, 6))
    plt.barh(feature_importances_df['Feature'], feature_importances_df['Importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance')
    plt.savefig('Visualisations/feature_importance_xgboost.png')

    filename = 'Models/xgboost_model.pkl'
    joblib.dump(xgb_clf, filename)


# Exuction order of code
def train_models(dataframe):
    X_train, X_test, y_train, y_test = data_preparation(dataframe)
    rf_hyperparameters, dt_hyperparameters, xgb_hyperparameters = hyperparameter_tuning(X_train, y_train)
    train_decision_tree(X_train, y_train, X_test, y_test, dt_hyperparameters)
    train_random_forest(X_train, y_train, X_test, y_test, rf_hyperparameters)
    train_xgboost(X_train, y_train, X_test, y_test, xgb_hyperparameters)

def predict_participants(dataframe, model):
    # Loading the model from joblib
    loaded_model = joblib.load(model)

    # Preparing the data for prediction
    dataframe = pd.get_dummies(dataframe, columns=['Medlemstype'], drop_first=True)
    dataframe = pd.get_dummies(dataframe, columns=['Region'], drop_first=True)
    dataframe['Aktiv_Deltager'] = dataframe['Aktiv_Deltager'].map({'Ja': 1, 'Nej': 0})

    kontakt_ids = dataframe['Kontakt_ID']
    features = dataframe.drop(columns=['Aktiv_Deltager', 'Kontakt_ID', 'Kontakt_OK', 'Antal_Aktiv', 'Status_Aarsag', 'Status', 'Startdato', 'Medlem_Status', 'Antal'])

    # Making predictions
    predictions = loaded_model.predict(features)
    probabilities = loaded_model.predict_proba(features)[:, 1]

    results = pd.DataFrame({
        'Kontakt_ID': kontakt_ids, 
        'Resultat': predictions, 
        'Sandsynlighed': probabilities}
    )
    results['Gruppe'] = pd.cut(results['Sandsynlighed'], bins=[0, 0.5, 0.7, 1], labels=['Lav', 'Medium', 'Høj'])
    results['Gruppe'] = results['Gruppe'].astype(str)

    results.to_csv('Results/predictions.csv', index=False)
    print("Predictions saved to 'Results/predictions.csv'")


#train_models(df) # Run this line to re-train the models on updated data
predict_participants(df, 'Models/random_forest_model.pkl') # decsion_tree_model.pkl or xgboost_model.pkl (optional) based on the model you want to use
    