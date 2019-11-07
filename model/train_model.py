import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.linear_model import LinearRegression
import lightgbm as lgb
import optuna
from sklearn.ensemble import GradientBoostingRegressor


def print_metrics(y_true, y_predicted):
    ## First compute R^2 and the adjusted R^2
    r2 = r2_score(y_true, y_predicted)

    ## Print the usual metrics and the R^2 values
    print('Mean Square Error      = ' + str(mean_squared_error(y_true, y_predicted)))
    print('Root Mean Square Error = ' + str(np.sqrt(mean_squared_error(y_true, y_predicted))))
    print('Mean Absolute Error    = ' + str(mean_absolute_error(y_true, y_predicted)))
    print('Median Absolute Error  = ' + str(median_absolute_error(y_true, y_predicted)))
    print('R^2                    = ' + str(r2))


#df = joblib.load('data/processed/training_data.pkl')
df = joblib.load('data/processed/training_data_state.pkl')
features = df.drop('rate_spread', axis=1)
labels = df[['row_id', 'rate_spread']]

X_train, X_test, y_train, y_test = train_test_split(features.drop('row_id', axis=1),
                                                    labels.rate_spread,
                                                    stratify=labels.rate_spread,
                                                    test_size=0.3)
print(y_train.value_counts(normalize=True))
print(y_test.value_counts(normalize=True))

xgb_model = xgb.XGBRegressor()
xgb_model.fit(X_train, y_train)
xgb_predictions = xgb_model.predict(X_test)
for i in range(len(xgb_predictions)):
    if xgb_predictions[i] < 1:
        xgb_predictions[i] = 1

print(r2_score(y_test, xgb_predictions))        # 0.7724 with state, 0.7711 without state (so pretty much the same)

test = pd.DataFrame({'real': y_test, 'predict': xgb_predictions})


lin_model = LinearRegression()
#scaler = StandardScaler()
#scaler.fit_transform(X_train)
lin_model.fit(X_train, y_train)
#scaler.transform(X_test)
lin_predictions = lin_model.predict(X_test)

print(r2_score(y_test, lin_predictions))

gbt = GradientBoostingRegressor(n_estimators=300, max_depth=2, random_state=42)
gbt.fit(X_train, y_train)
gbt_preds = gbt.predict(X_test)
print(r2_score(y_test, gbt_preds))

# Save models
joblib.dump(xgb_model, 'data/models/xgb_state_07685.pkl')


# HYPER PARAMETER TUNING XGBOOST
from sklearn.model_selection import GridSearchCV
hxgb = xgb.XGBRegressor(nthread=-1)
params = {
    'max_depth': range(2, 4, 1),
    'gamma': [0.2, 0.4, 0.6],
    'min_child_weight': [4, 5],
    'learning_rate': [0.1, 0.01, 0.05]
}
grid = GridSearchCV(hxgb, params, verbose=True, n_jobs=3, scoring='r2')
grid.fit(X_train, y_train)
