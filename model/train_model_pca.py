from sklearn.decomposition import TruncatedSVD
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression

df = joblib.load('data/processed/training_data_pca.pkl')

# Change dtype of the categoricals
df['msa_md'] = df.msa_md.astype('category')
df = pd.concat([df, pd.get_dummies(df['msa_md'], 'msa')], axis=1)
df = df.drop('msa_md', axis=1)
df['county_code'] = df.county_code.astype('category')
df = pd.concat([df, pd.get_dummies(df['county_code'], 'county')], axis=1)
df = df.drop('county_code', axis=1)
df['lender'] = df.lender.astype('category')
df = pd.concat([df, pd.get_dummies(df['lender'], 'lender')], axis=1)
df = df.drop('lender', axis=1)
df = df.drop('state_code', axis=1)
df = df[df.rate_spread < 17]
features = df.drop('rate_spread', axis=1)
labels = df[['row_id', 'rate_spread']]

# Train model on this mess
X_train, X_test, y_train, y_test = train_test_split(features.drop('row_id', axis=1),
                                                    labels.rate_spread,
                                                    stratify=labels.rate_spread,
                                                    test_size=.3)

#pca = TruncatedSVD(n_components=400, n_iter=7, random_state=42)
#X_train = pca.fit_transform(X_train)
lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
#X_test = pca.transform(X_test)
lin_preds = lin_model.predict(X_test)
print(r2_score(y_test, lin_preds))      # 0.716
joblib.dump(lin_preds, 'data/models/lin_all_vars_0716.pkl')

xgb_model = XGBRegressor()
xgb_model.fit(X_train, y_train)

xgb_preds = xgb_model.predict(X_test)
print(r2_score(y_test, xgb_preds))
joblib.dump(lin_preds, 'data/models/xgb_all_vars_0716.pkl')
