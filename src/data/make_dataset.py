import pandas as pd
import joblib
import numpy as np

# Load the slightly preprocessed data
df = pd.read_csv('data/raw/train_values.csv')

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

# For state code, we will simply ad a missing state
df.state_code.describe()            # Missing values, max 52 (number of US states, duh)
df['state_code'] = df.state_code.apply(lambda x: 53 if x == -1 else x)

# Drop lender since we grouped them already
df = df.drop('lender', axis=1)

# Create loan income q
df['loan_income_q'] = df['applicant_income'] / df['loan_amount']

# Remove applicant_income and replace with log version
df['applicant_income_log'] = df['applicant_income'].apply(lambda x: np.log(x))
df = df.drop('applicant_income', axis=1)

# Since we created the loan amount income quote, we remove loan amount
df = df.drop('loan_amount', axis=1)

# Since msa_md, state_code and county_code are large but varying, we drop msa, and county since they are not unique
df = df.drop('msa_md', axis=1)
df = df.drop('county_code', axis=1)

# Population - create a median fill
df.population = df.population.fillna(df.population.median())

# Number of 1 to 4 family units
df['number_of_1_to_4_family_units'] = df['number_of_1_to_4_family_units'].fillna(df['number_of_1_to_4_family_units'].median())

# Create new feature
df['small_house_pct'] = df['population'] / df['number_of_1_to_4_family_units']

# Empty house and overcrowded
df['empty_overcrowd'] = df['number_of_owner-occupied_units'] / df['population']

# Create Categorical values
cat_features = [feature for feature in df.columns if df[feature].dtypes == 'object']

for cat in cat_features:
    df = pd.concat([df, pd.get_dummies(df[cat], cat)], axis=1)
    df = df.drop(cat, axis=1)
