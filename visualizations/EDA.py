import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


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

# Property Location (here -1 is a missing value for some reason)
df.msa_md.value_counts()            # 409 different values, but no missing values in training set
df.msa_md.describe()                # No missing values! min = 0
df.state_code.describe()            # Missing values, max 52 (number of US states, duh)
len(df[df['state_code'] == -1])     # 1338 missing values
df.county_code.describe()           # No missing values in training set, 316 different counties


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
df.groupby('loan_type')['rate_spread'].value_counts(normalize=True).unstack().plot(kind='bar', stacked=True, ax=ax1, legend=False)
plt.title('Loan type')
plt.xticks(rotation=45)
ax2 = fig.add_subplot(132)
df.groupby('loan_purpose')['rate_spread'].value_counts(normalize=True).unstack().plot(kind='bar', stacked=True, ax=ax2, legend=False)
plt.title('Loan Purpose')
plt.xticks(rotation=45)
ax3 = fig.add_subplot(133)
df.groupby('lender_group')['rate_spread'].value_counts(normalize=True).unstack().plot(kind='bar', stacked=True, ax=ax3, legend=False)
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


