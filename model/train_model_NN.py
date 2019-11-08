import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from keras.layers import Dense, Dropout
from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD

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

m, n = X_train.shape        # Get the shapes for building the network
scaler = StandardScaler()   # Most likely we need to scale the input
Xnp_train = scaler.fit_transform(X_train)
Xs_test = scaler.transform(X_test)
ynp_train = pd.get_dummies(y_train).values

# First Model
model = Sequential()
model.add(Dense(n, activation='relu', input_shape=(n, )))
model.add(Dense(2*n, activation='relu'))
model.add(Dense(n, activation='relu'))
model.add(Dense(16, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
early_stopping_monitor = EarlyStopping(patience=2)

model.fit(Xnp_train, ynp_train, epochs=30, callbacks=[early_stopping_monitor])

# Evaluate
nn_preds = model.predict(Xs_test)
nn_preds_clean = []
nn_preds_probs = []
for i in range(len(nn_preds)):
    nn_preds_clean.append(np.argmax(nn_preds[i, :]) + 1)
    nn_preds_probs.append(nn_preds[i, np.argmax(nn_preds[i, :])])

print(r2_score(y_test, nn_preds_clean))

# Second Model
model2 = Sequential()
model2.add(Dense(n, activation='relu', input_shape=(n, )))
model2.add(Dropout(0.5))
model2.add(Dense(n, activation='relu'))
model2.add(Dropout(0.5))
model2.add(Dense(16, activation='softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=.9, nesterov=True)
model2.compile(loss='categorical_crossentropy',
               optimizer=sgd,
               metrics=['accuracy'])

model2.fit(Xnp_train, ynp_train, epochs=30, batch_size=128)

# Evaluate second model
score = model2.evaluate(Xs_test, pd.get_dummies(y_test).values)
nn_preds = model2.predict(Xs_test)
nn_preds_clean = []
nn_preds_probs = []
for i in range(len(nn_preds)):
    nn_preds_clean.append(np.argmax(nn_preds[i, :]) + 1)
    nn_preds_probs.append(nn_preds[i, np.argmax(nn_preds[i, :])])

print(r2_score(y_test, nn_preds_clean))
