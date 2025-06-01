import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Muat Dataset
df = pd.read_csv('./loan_data.csv')

# Kolom Numerikal
numerical_column = ['person_age', 'person_income', 'person_emp_exp', 'loan_amnt',
                     'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length', 'credit_score']

# Atasi Missing Value & Duplicated
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# Atasi Outlier
Q1 = df[numerical_column].quantile(0.25)
Q3 = df[numerical_column].quantile(0.75)
IQR = Q3 - Q1

df = df[~((df[numerical_column] < (Q1-1.5*IQR)) | (df[numerical_column] > (Q3+1.5*IQR))).any(axis=1)]

# Encoding
## Binary Encoding
df['person_gender'] = df['person_gender'].map({'male':1, 'female':0})
df['previous_loan_defaults_on_file'] = df['previous_loan_defaults_on_file'].map({'Yes':1, 'No':0})

## Mapping Educational column
df['person_education'] = df['person_education'].map(
    {
        'High School': 0,
        'Associate': 1,
        'Bachelor': 2,
        'Master': 3,
        'Doctorate': 4
    }
)

## One Hot Encoding
df_encoded = df.copy()
df_encoded = pd.get_dummies(df, columns=['loan_intent', 'person_home_ownership'])

# Standardisasi
scaler = StandardScaler()
df_encoded[numerical_column] = scaler.fit_transform(df_encoded[numerical_column])

# Save Dataset
df_cleaned = df_encoded.copy()
df_cleaned.to_csv("preprocessing/loan_data_preprocessing.csv",index=False)