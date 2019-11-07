import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
import joblib

def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    # Number of data points: n
    n = len(data)

    # x-data for the ECDF: x
    x = np.sort(data)

    # y-data for the ECDF: y
    y = np.arange(1, n+1) / n

    return x, y


# Filepaths
features_fp = 'data/raw/train_values.csv'
labels_fp = 'data/raw/train_labels.csv'

# Read files
features = pd.read_csv(features_fp) # The original which we won't touch
labels = pd.read_csv(labels_fp)     # The training labels
df = pd.read_csv(features_fp)   # The copy which we will adjust
df = pd.merge(left=df, right=labels, on='row_id', how='left')
#df = joblib.load('data/interim/df.pkl')
labels = df[['row_id', 'rate_spread']]

# Basic info
print(features.shape)   # (2000000, 20)
print(features.columns) # 20 different
print(features.dtypes)  #

# Analyze each feature
df.lender.value_counts()    # Gives 3893 different values, with max, min = 9235, 1
df.loan_amount.describe()   # mean 142, std 142, (25th, 50th, 75th) = (67, 116, 179)
df.loan_type.value_counts() # Must FHA-insured, least VA-guaranteed
df.property_type.value_counts()
df.loan_purpose.value_counts()
df.occupancy.value_counts()
df.preapproval.value_counts()
_ = plt.hist(x=df['applicant_income'].apply(lambda x: np.log(x)), bins=50)  # Pretty normal distributed

df.applicant_ethnicity.value_counts()

df.applicant_race.value_counts()    # predominantly white, 2nd black and 3rd info not provided

rs_x, rs_y = ecdf(labels.rate_spread)
_ = plt.plot(rs_x, rs_y)                        # Majority seem to be close to 1
print(np.percentile(labels.rate_spread, 57.5))    # All 1 at 57.5 percentile
_ = plt.hist(np.log(labels.rate_spread), bins=10)

# Boxplot to see how the outliers in the rate spread are behaviouring
_ = sns.boxplot(y='rate_spread', data=labels)       # we should remove ratespread greater than 20

# Property Location (here -1 is a missing value for some reason)
df.msa_md.value_counts()            # 409 different values, but no missing values in training set
df.msa_md.describe()                # No missing values! min = 0
df.state_code.describe()            # Missing values, max 52 (number of US states, duh)
len(df[df['state_code'] == -1])     # 1338 missing values
df.county_code.describe()           # No missing values in training set, 316 different counties

# Effect of lender on rate spread
tt = df[['row_id', 'lender', 'rate_spread']]
lender_max = dict()
lender_mean = dict()
llenders = tt.lender.unique().tolist()
for lender in llenders:
    lender_max[lender] = max(tt[tt['lender'] == lender]['rate_spread'])
    lender_mean[lender] = np.mean(tt[tt['lender'] == lender]['rate_spread'])

lender, max_rate = zip(*lender_max.items())
lender2, mean_rate = zip(*lender_mean.items())
plt.subplot(2, 1, 1)
_ = plt.hist(max_rate, bins=50)
plt.title('Max rate')
plt.subplot(2, 1, 2)
_ = plt.hist(mean_rate, bins=50)
plt.title('Mean rate')

tt['max_spread'] = tt['rate_spread']

# Feature engineering - Numerical
df['loan_income_q'] = df['loan_amount'] / df['applicant_income']

# Investigate numerical correlation and categorical variance
corr = df.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})

# Feature engineering - Categorical
# We want to divide the lenders into 4 buckets: tiny, small, medium, large depending on how many
# loans they give out.
df.lender.value_counts()    # Max around 9000, then dropping of with a couple in the thousands and plenty in 1

# Variance analysis in categorical features
cat_features = [feature for feature in df.columns if df[feature].dtypes == 'object']

fig = plt.figure()
fig.suptitle('Loan Categories')
ax1 = fig.add_subplot(131)
df.groupby('loan_type')['rate_spread'].value_counts(normalize=True).unstack().plot(kind='bar', stacked=True, ax=ax1,
                                                                                   legend=False)
plt.title('Loan type')
plt.xticks(rotation=45)
ax2 = fig.add_subplot(132)
df.groupby('loan_purpose')['rate_spread'].value_counts(normalize=True).unstack().plot(kind='bar', stacked=True, ax=ax2,
                                                                                      legend=False)
plt.title('Loan Purpose')
plt.xticks(rotation=45)
ax3 = fig.add_subplot(133)
df.groupby('lender_group')['rate_spread'].value_counts(normalize=True).unstack().plot(kind='bar', stacked=True, ax=ax3,
                                                                                      legend=False)
plt.title('Lender Group')
plt.xticks(rotation=45)
handles, labs = ax1.get_legend_handles_labels()
fig.legend(handles, labs, loc='best')

fig2 = plt.figure()
fig2.suptitle('Applicant Categories')
ax1 = fig2.add_subplot(131)
df.groupby('applicant_ethnicity')['rate_spread'].value_counts(normalize=True).unstack().plot(kind='bar', stacked=True, ax=ax1, legend=False)
plt.title('Applicant Ethnicity')
plt.xticks(rotation=45)
ax2 = fig2.add_subplot(132)
df.groupby('applicant_race')['rate_spread'].value_counts(normalize=True).unstack().plot(kind='bar', stacked=True, ax=ax2, legend=False)
plt.title('Applicant Race')
plt.xticks(rotation=45)
ax3 = fig2.add_subplot(133)
df.groupby('applicant_sex')['rate_spread'].value_counts(normalize=True).unstack().plot(kind='bar', stacked=True, ax=ax3, legend=False)
plt.title('Applicant Sex')
plt.xticks(rotation=45)
handles, labs = ax1.get_legend_handles_labels()
fig2.legend(handles, labs, loc='best')

fig3 = plt.figure()
fig3.suptitle('Bank Categories')
ax1 = fig3.add_subplot(131)
df.groupby('occupancy')['rate_spread'].value_counts(normalize=True).unstack().plot(kind='bar', stacked=True, ax=ax1, legend=False)
plt.title('Occupancy')
plt.xticks(rotation=45)
ax2 = fig3.add_subplot(132)
df.groupby('preapproval')['rate_spread'].value_counts(normalize=True).unstack().plot(kind='bar', stacked=True, ax=ax2, legend=False)
plt.title('Preapproval')
plt.xticks(rotation=45)
ax3 = fig3.add_subplot(133)
df.groupby('property_type')['rate_spread'].value_counts(normalize=True).unstack().plot(kind='bar', stacked=True, ax=ax3, legend=False)
plt.title('Property Type')
plt.xticks(rotation=45)
handles, labs = ax1.get_legend_handles_labels()
fig3.legend(handles, labs, loc='best')

import joblib
joblib.dump(df, 'data/interim/df.pkl')

# Missing values
df_missing = df[df['population'].isnull()]  # Missing values are pretty much in the same rows, so probably one
df_missing.county_code.value_counts()       # Majority belong to county code 316

# Check the effect of state
tt = df[['row_id', 'state_code', 'rate_spread']]
# Remove outliers from the training dataset
tt = tt[tt.rate_spread < 17]
state_max = dict()
state_mean = dict()
states = tt.state_code.unique().tolist()
for state in states:
    state_max[state] = max(tt[tt['state_code'] == state]['rate_spread'])
    state_mean[state] = np.mean(tt[tt['state_code'] == state]['rate_spread'])

state_code, max_rate = zip(*state_max.items())
state_code2, mean_rate = zip(*state_mean.items())

plt.subplot(2, 1, 1)
plt.title('Max Rate')
_ = plt.bar(state_code, max_rate)
plt.subplot(2, 1, 2)
plt.title('Mean Rate')
_ = plt.bar(state_code2, mean_rate)

# Compare training and test dataset
test = pd.read_csv('data/raw/test_values.csv')
train = pd.read_csv('data/raw/train_values.csv')

print(len(test.lender.unique()))        # 3900
print(len(train.lender.unique()))       # 3893
i = 0
for lender in train.lender.unique():
    if lender in test.lender.unique():
        i += 1                          # Will output 3500, meaning there are

