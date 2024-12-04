import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def replace_age_with_age_group(df:pd.DataFrame):
    if df['age'] < 18:
        return 'teenagers'
    elif df['age'] >= 18 and df['age'] < 30:
        return 'young adults'
    elif df['age'] >= 30 and df['age'] < 45:
        return 'middle-aged adults'
    elif df['age'] >= 45:
        return 'old-aged adults'
    
    
def return_train_test_split(adult_df:pd.DataFrame):
    adult_df.replace("?", np.nan, inplace=True)
    adult_df.dropna(inplace=True, ignore_index=True)
    adult_df['>50K,<=50K'].replace(['<=50K', '>50K'], [0, 1], inplace=True)
    adult_df['age_groups'] = adult_df.apply(replace_age_with_age_group, axis=1)
    cols = [x for x in adult_df.columns if x != '>50K,<=50K']
    cols_for_dummies = ['workclass', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'age_groups']
    data = adult_df[cols]
    target = adult_df['>50K,<=50K']
    data = pd.get_dummies(data, columns=cols_for_dummies, dtype=int)
    X_train, X_test, y_train, y_test = train_test_split(data, target, train_size=0.6, random_state=1)
    
    return X_train, X_test, y_train, y_test