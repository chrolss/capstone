import pandas as pd
import numpy as np
import joblib

#df = joblib.load('data/processed/test_data.pkl')
df = joblib.load('data/processed/test_data_state.pkl')
train = joblib.load('data/processed/training_data_state.pkl')
train = train.drop('rate_spread', axis=1)

df = df[train.columns]

model = joblib.load('data/models/xgb_state_07685.pkl')

preds = model.predict(df.drop('row_id', axis=1))

output = pd.DataFrame({'row_id': df.row_id, 'rate_spread': preds})

output.to_csv('submission191107b.csv', index=False)
