import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_log_error
import lightgbm as lgb
from xgboost import XGBRegressor

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')


train_df['Item_Weight'].fillna(train_df.groupby('Item_Identifier')['Item_Weight'].transform('mean'), inplace=True)
test_df['Item_Weight'].fillna(test_df.groupby('Item_Identifier')['Item_Weight'].transform('mean'), inplace=True)
train_df['Outlet_Size'].fillna(train_df['Outlet_Size'].mode()[0], inplace=True)
test_df['Outlet_Size'].fillna(test_df['Outlet_Size'].mode()[0], inplace=True)


train_df['Item_Visibility_Squared'] = train_df['Item_Visibility'] ** 2
test_df['Item_Visibility_Squared'] = test_df['Item_Visibility'] ** 2
train_df['Item_Type_Combined'] = train_df['Item_Identifier'].apply(lambda x: x[0:2])
test_df['Item_Type_Combined'] = test_df['Item_Identifier'].apply(lambda x: x[0:2])
train_df['Outlet_Years'] = 2024 - train_df['Outlet_Establishment_Year']
test_df['Outlet_Years'] = 2024 - test_df['Outlet_Establishment_Year']
train_df['Price_Per_Weight'] = train_df['Item_MRP'] / train_df['Item_Weight']
test_df['Price_Per_Weight'] = test_df['Item_MRP'] / test_df['Item_Weight']


train_df['Item_MRP_Log'] = np.log1p(train_df['Item_MRP'])
test_df['Item_MRP_Log'] = np.log1p(test_df['Item_MRP'])
train_df['Item_Visibility_Log'] = np.log1p(train_df['Item_Visibility'])
test_df['Item_Visibility_Log'] = np.log1p(test_df['Item_Visibility'])


le = LabelEncoder()
categorical_cols = ['Item_Identifier', 'Item_Fat_Content', 'Item_Type', 'Outlet_Identifier', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type', 'Item_Type_Combined']
for col in categorical_cols:
    train_df[col] = le.fit_transform(train_df[col])
    test_df[col] = le.transform(test_df[col])


numerical_cols = ['Item_Weight', 'Item_Visibility', 'Item_MRP', 'Item_Visibility_Squared', 'Outlet_Years', 'Price_Per_Weight', 'Item_MRP_Log', 'Item_Visibility_Log']
scaler = StandardScaler()
train_df[numerical_cols] = scaler.fit_transform(train_df[numerical_cols])
test_df[numerical_cols] = scaler.transform(test_df[numerical_cols])


X = train_df.drop(columns=['id', 'Item_Outlet_Sales'])
y = np.log1p(train_df['Item_Outlet_Sales'])  
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


xgb_model = XGBRegressor(random_state=42)
xgb_param_grid = {
    'n_estimators': [600, 1000, 1500],
    'learning_rate': [0.01, 0.03, 0.05],
    'max_depth': [5, 8, 10],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9]
}

xgb_search = RandomizedSearchCV(estimator=xgb_model, param_distributions=xgb_param_grid, cv=3, n_iter=20, scoring='neg_mean_squared_log_error', n_jobs=-1, verbose=1, random_state=42)
xgb_search.fit(X_train, y_train)
xgb_best_model = xgb_search.best_estimator_


y_pred_xgb = xgb_best_model.predict(X_val)
rmsle_xgb = np.sqrt(mean_squared_log_error(np.expm1(y_val), np.expm1(y_pred_xgb)))  
print('XGBoost Validation RMSLE:', rmsle_xgb)
print('XGBoost Best Parameters:', xgb_search.best_params_)

lgb_model = lgb.LGBMRegressor(random_state=42)
lgb_param_grid = {
    'num_leaves': [31, 50, 70],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [1000, 1500, 2000],
    'max_depth': [-1, 5, 10],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9]
}

lgb_search = RandomizedSearchCV(estimator=lgb_model, param_distributions=lgb_param_grid, cv=3, n_iter=20, n_jobs=-1, scoring='neg_mean_squared_log_error', verbose=1, random_state=42)
lgb_search.fit(X_train, y_train)
lgb_best_model = lgb_search.best_estimator_

y_pred_lgb = lgb_best_model.predict(X_val)
rmsle_lgb = np.sqrt(mean_squared_log_error(np.expm1(y_val), np.expm1(y_pred_lgb)))  
print('LightGBM Validation RMSLE:', rmsle_lgb)
print('LightGBM Best Parameters:', lgb_search.best_params_)

final_preds = (np.expm1(y_pred_xgb) + np.expm1(y_pred_lgb)) / 2
final_rmsle = np.sqrt(mean_squared_log_error(np.expm1(y_val), final_preds))
print('Ensemble Validation RMSLE:', final_rmsle)

X_test = test_df.drop(columns=['id'])
test_preds_xgb = np.expm1(xgb_best_model.predict(X_test))
test_preds_lgb = np.expm1(lgb_best_model.predict(X_test))
final_test_preds = (test_preds_xgb + test_preds_lgb) / 2

submission = pd.DataFrame({
    'id': test_df['id'],
    'Item_Outlet_Sales': final_test_preds
})

submission.to_csv('submission.csv', index=False)