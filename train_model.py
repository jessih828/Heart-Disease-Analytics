import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from collections import Counter 
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, classification_report
import pickle


df = pd.read_csv('Heart Attack Analytics/heart_data.csv')
object_col = df.select_dtypes(include='object').columns
df = pd.get_dummies(df, columns=object_col, drop_first=True)

X = df.drop(['HeartDisease'], axis=1)
y = df['HeartDisease']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size= 0.2, 
    stratify=y,
    random_state=123)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()  
X_train_scaled = scaler.fit_transform(X_train)  
X_test_scaled = scaler.transform(X_test) 

xgb_grid = XGBClassifier(n_estimators=200, use_label_encoder=False, eval_metric='logloss')

param_grid = {
    'subsample': [0.6, 0.7, 1.0],
    'max_depth': [2, 3, 5, 7, 9], 
    'learning_rate': [0.001, 0.01, 0.05, 0.1],
    'min_child_weight': [1, 3, 5],
    'colsample_bytree': [0.3, 0.5, 0.7, 1.0],
    'gamma': [0, 0.1, 0.3, 0.5]
}

# Initialize GridSearchCV with StratifiedKFold
cv_strategy = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
grid_search = GridSearchCV(estimator=xgb_grid, param_grid=param_grid, scoring='accuracy', cv=cv_strategy, n_jobs=-1, verbose=3)

grid_search.fit(X_train, y_train)

# Print the best parameters and the corresponding score
print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

best_params = grid_search.best_params_
xgboost_model = XGBClassifier(n_estimators=200, 
                            subsample=best_params["subsample"],
                            max_depth=best_params["max_depth"],
                            learning_rate=best_params["learning_rate"],
                            min_child_weight=best_params["min_child_weight"],
                            colsample_bytree=best_params["colsample_bytree"],
                            gamma=best_params["gamma"],
                            use_label_encoder=False, eval_metric='logloss')
xgboost_model.fit(X_train, y_train)

# Save the trained model to a pickle file
with open('heart_attack_model.pkl', 'wb') as file:
    pickle.dump(xgboost_model, file)