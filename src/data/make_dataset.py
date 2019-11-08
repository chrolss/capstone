import pandas as pd
import joblib
import numpy as np

# Load the slightly preprocessed data
df = pd.read_csv('data/raw/train_values.csv')
labels = pd.read_csv('data/raw/train_labels.csv')

def make_dataset():
    print("hejsan")

# Map categoricals
df['loan_type'] = df.loan_type.map({1: 'Conventional', 2: 'FHA-insured', 3: 'VA-guaranteed', 4: 'FSA/RHS'})
df['property_type'] = df.property_type.map({1: 'One to four-family', 2: 'Manufactured housing', 3: 'Multifamily'})
df['loan_purpose'] = df.loan_purpose.map({1: 'Home purchase', 2: 'Not owner-occupied', 3: 'Not applicable'})
df['occupancy'] = df.occupancy.map({1: 'Owner-occupied as principal dwelling', 2: 'Not owner-occupied', 3: 'Not applicable'})
df['preapproval'] = df.preapproval.map({1: 'Preapproval was requested', 2: 'Preapproval was not requested', 3: 'Not applicable'})
df['applicant_ethnicity'] = df.applicant_ethnicity.map({1: 'Hispanic or Latino',
                                                        2:'Not hispanic or latino',
                                                        3: 'Info not provided',
                                                        4: 'Not applicable',
                                                        5: 'No co-applicant'})
df['applicant_race'] = df.applicant_race.map({
    1: 'American Indian or Alaska Native',
    2: 'Asian',
    3: 'Black or African American',
    4: 'Native Hawaiian or Other pacific islander',
    5: 'White',
    6: 'Info not provided',
    7: 'Not applicable',
    8: 'No co-applicant'
})
df['applicant_sex'] = df.applicant_sex.map({
    1: 'Male',
    2: 'Female',
    3: 'Info not provided',
    4: 'Not applicable',
    5: 'Not applicable'
})

lenders = df.lender.value_counts().to_dict()
lender_group = dict()
for lender in lenders.keys():
    temp = lenders[lender]
    if temp > 1000:
        lender_group[lender] = 'large'
    elif temp > 500:
        lender_group[lender] = 'medium'
    elif temp > 20:
        lender_group[lender] = 'small'
    else:
        lender_group[lender] = 'tiny'


df['lender_group'] = df.lender.apply(lambda x: lender_group[x])
joblib.dump(lender_group, 'data/models/lender_group.pkl')
# Statistics on the lenders
tt = pd.merge(df, labels, on='row_id')
lender_max = dict()
lender_mean = dict()
llenders = df.lender.unique().tolist()
for lender in llenders:
    lender_max[lender] = max(tt[tt['lender'] == lender]['rate_spread'])
    lender_mean[lender] = np.mean(tt[tt['lender'] == lender]['rate_spread'])

df['lender_mean'] = df.lender.apply(lambda x: lender_mean[x])
df['lender_max'] = df.lender.apply(lambda x: lender_max[x])
joblib.dump(lender_mean, 'data/models/lender_mean.pkl')
joblib.dump(lender_max, 'data/models/lender_max.pkl')

# Fill missing values
df.isnull().sum()   # Applicant income is the greatest missing value, eather median fill or model for it?
# population 1% missing
# minority_population_pct           1% missing
# ffiecmedian_family_income         1% missing
# tract_to_msa_md_income_pct        1% missing
# number_of_owner-occupied_units    10% missing
# number_of_1_to_4_family_units     10% missing

# Income - fill with median
df.applicant_income = df.applicant_income.fillna(df.applicant_income.median())
df['ffiecmedian_family_income'] = df.ffiecmedian_family_income.fillna(df.ffiecmedian_family_income.median())
df['tract_to_msa_md_income_pct'] = df.tract_to_msa_md_income_pct.fillna(df.tract_to_msa_md_income_pct.median())

# For state code, we will simply ad a missing state
df.state_code.describe()            # Missing values, max 52 (number of US states, duh)
df['state_code'] = df.state_code.apply(lambda x: 53 if x == -1 else x)

# Create loan income q
df['loan_income_q'] = df['applicant_income'] / df['loan_amount']

# Remove applicant_income and replace with log version
df['applicant_income_log'] = df['applicant_income'].apply(lambda x: np.log(x))

# Population - create a median fill
df.population = df.population.fillna(df.population.median())

# Minority_population_pct - median fill
df.minority_population_pct = df.minority_population_pct.fillna(df.minority_population_pct.median())

# Number of 1 to 4 family units
df['number_of_1_to_4_family_units'] = df['number_of_1_to_4_family_units'].fillna(
    df['number_of_1_to_4_family_units'].median())
df['number_of_owner-occupied_units'] = df['number_of_owner-occupied_units'].fillna(
    df['number_of_owner-occupied_units'].median())

# Create new feature
df['small_house_pct'] = df['population'] / df['number_of_1_to_4_family_units']

# Empty house and overcrowded
df['empty_overcrowd'] = df['number_of_owner-occupied_units'] / df['population']

# Applicants income vs median pct
df['income_median_pct'] = df['applicant_income'] / df['ffiecmedian_family_income']

# Create Categorical values
cat_features = [feature for feature in df.columns if df[feature].dtypes == 'object']

for cat in cat_features:
    df = pd.concat([df, pd.get_dummies(df[cat], cat, drop_first=True)], axis=1)
    df = df.drop(cat, axis=1)

df['state_code'] = df.state_code.astype('category')
df = pd.concat([df, pd.get_dummies(df.state_code, 'state')], axis=1)

# Perform logs on large values
df['ffiecmedian_log'] = df['ffiecmedian_family_income'].apply(lambda x: np.log(x))
df['number_of_1_to_4_family_units_log'] = df['number_of_1_to_4_family_units'].apply(lambda x: np.log(x))
df['population_log'] = df['population'].apply(lambda x: np.log(x))

# Remove outliers from the training dataset
df = pd.merge(left=df, right=labels, on='row_id', how='left')
df = df[df.rate_spread < 17]

# Save the big dataframe
joblib.dump(df, 'data/processed/training_data_pca.pkl')

# Drop unnecessary values
df = df.drop('applicant_income', axis=1)    # We have the log of this instead
df = df.drop('loan_amount', axis=1)         # Since we created the loan amount income quote, we remove loan amount
df = df.drop('msa_md', axis=1)              # Since msa_md and county_code are large but varying
df = df.drop('county_code', axis=1)
df = df.drop('number_of_owner-occupied_units', axis=1)
df = df.drop('number_of_1_to_4_family_units', axis=1)
df = df.drop('ffiecmedian_family_income', axis=1)
df = df.drop('population', axis=1)
df = df.drop('state_code', axis=1)
df = df.drop('lender', axis=1)              # Drop lender since we grouped them already

joblib.dump(df, 'data/processed/training_data.pkl')

# Appendix
tt = pd.merge(left=df, right=labels, on='row_id', how='left')
import matplotlib.pyplot as plt
import seaborn as sns
corr = tt.corr()
mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize=(11, 9))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0, square=True, linewidths=.5, cbar_kws={"shrink": .5})


