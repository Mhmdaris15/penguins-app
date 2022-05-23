import pandas as pd
import streamlit as st

penguin_data = pd.read_csv('data/penguins_cleaned.csv')

st.write(penguin_data)

encode_col = ['island', 'sex']

for col in encode_col:
    dummy = pd.get_dummies(penguin_data[col], prefix=col)
    penguin_data = pd.concat([penguin_data, dummy], axis=1)
    penguin_data.drop(columns=[col], inplace=True)
    # del penguin_data[col]

st.write(penguin_data)

target_mapper = {'Adelie':0, 'Chinstrap':1, 'Gentoo':2}

penguin_data['species'] = penguin_data['species'].apply(lambda x: target_mapper[x])

# Split Data
X = penguin_data.drop(columns='species')
y = penguin_data['species']

# Train the model 
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X, y)

# Saving the model
import pickle
pickle.dump(clf, open('./data/penguin_clf_220522.pkl', 'wb'))