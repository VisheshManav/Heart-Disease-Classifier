import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler

import pickle

df = pd.read_csv('heart_cleveland_upload.csv')

numerical = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
categorical = ['sex', 'fbs', 'exang', 'condition', 'cp', 'restecg', 'slope', 'ca', 'thal']

cat_map_to_string = {
    'sex'       : ['female', 'male'],
    'fbs'       : ['false', 'true'],
    'exang'     : ['no', 'yes'], 
    'condition' : ['no_disease', 'disease'], 
    'cp'        : ["typical_angina", "atypical_angina", "non_anginal_pain", "asymtomatic"], 
    'restecg'   : ['normal', 'st-t_wave_abnormality', 'left_ventricular_hypertrophy'], 
    'slope'     : ['upsloping', 'flat', 'downsloping'], 
    'ca'        : ['zero', 'one', 'two', 'three'], 
    'thal'      : ['normal', 'fixed_defect', 'reversable_defect']
}

for cat in categorical:
    c = cat_map_to_string[cat]
    df[cat] = df[cat].map(lambda x: c[x])

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=49)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=49)

df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)

y_train = df_train.condition.values
y_val = df_val.condition.values
y_test = df_test.condition.values

del df_train['condition']
del df_val['condition']
del df_test['condition']

train_dicts = df_train.to_dict(orient='records')

dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(train_dicts)

full_train_dicts = df_full_train.to_dict(orient='records')
X_full_train = dv.transform(full_train_dicts)
y_full_train = df_full_train.condition.values

test_dicts = df_test.to_dict(orient='records')
X_test = dv.transform(test_dicts)

model_LR = LogisticRegression(random_state=49)
model_LR.fit(X_train, y_train)

val_dicts = df_val.to_dict(orient='records')
X_val = dv.transform(val_dicts)

y_pred = model_LR.predict_proba(X_val)[:, 1]

model_LR_1 = LogisticRegression(max_iter=400, C=0.1,
                            solver='lbfgs', random_state=49)
model_LR_1.fit(X_full_train, y_full_train)

output_file = 'model_lr.bin'
with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model_LR_1), f_out)

print(f'the model is saved to {output_file}')

max_depth = 4
min_samples_leaf = 1
n_estimators = 40
model_rf = RandomForestClassifier(n_estimators=n_estimators,
                            max_depth=max_depth, 
                            min_samples_leaf=min_samples_leaf, 
                            random_state=49)
model_rf.fit(X_full_train, y_full_train)

y_pred = model_rf.predict_proba(X_test)[:, 1]

output_file = 'model_rf.bin'
with open(output_file, 'wb') as f_out:
    pickle.dump((dv, model_rf), f_out)

print(f'The model is saved to {output_file}')