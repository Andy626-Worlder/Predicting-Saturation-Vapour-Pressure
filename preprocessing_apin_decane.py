#%%
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

def pre_apin_decane(train):
    selected_categories_for_apin_decane = ['apin', 'decane', 'apin_decane']
    apin_decane = train[train['parentspecies'].isin(selected_categories_for_apin_decane)]
    #print(apin_decane.head())

    #print(apin_decane['parentspecies'].value_counts())

    apin_decane['parentspecies'] = apin_decane['parentspecies'].replace({'apin_decane': np.nan})


    #print('apin_decane' in apin_decane['parentspecies'].values)

    #apin_decane_train, apin_decane_test=train_test_split(apin_decane, test_size=0.33, random_state=42)

    #print("apin_decane_train: ", apin_decane_train['parentspecies'].value_counts())
    #print("apin_decane_test: ", apin_decane_test['parentspecies'].value_counts())

    #print("a:",apin_decane['parentspecies'].isna().sum())

    df_missing_y = apin_decane[apin_decane['parentspecies'].isna()]
    #print(df_missing_y.head())

    df_no_missing_y = apin_decane.dropna(subset=['parentspecies'])

    #print(df_missing_y.head())
    #print(df_no_missing_y.head())

    # Separate features (X) and target variable (y) for the dataset without missing y
    X_train = df_no_missing_y.drop(columns=['parentspecies'])
    y_train = df_no_missing_y['parentspecies']


    X_missing_y = df_missing_y.drop(columns=['parentspecies'])

    knn_classifier = KNeighborsClassifier(n_neighbors=5)

    # Train the classifier on the dataset without missing y
    knn_classifier.fit(X_train, y_train)

    # Make predictions for the missing y values
    y_missing_pred = knn_classifier.predict(X_missing_y)

    #print(apin_decane['parentspecies'].value_counts())
    # Replace the missing y values with the predicted values
    apin_decane.loc[apin_decane['parentspecies'].isna(), 'parentspecies'] = y_missing_pred
    #print(apin_decane['parentspecies'].value_counts())

    for index, row in apin_decane.iterrows():
        id_match = row['Id']
        new_value = row['parentspecies']
        
        train.loc[train['Id'] == id_match, 'parentspecies'] = new_value

    #print(train['parentspecies'].value_counts())
    return train