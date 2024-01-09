#%%
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

def pre_apin_toluene(train):
    selected_categories_for_apin_toluene = ['apin', 'toluene', 'apin_toluene']
    apin_toluene = train[train['parentspecies'].isin(selected_categories_for_apin_toluene)]
    #print(apin_toluene.head())

    #print(apin_toluene['parentspecies'].value_counts())

    apin_toluene['parentspecies'] = apin_toluene['parentspecies'].replace({'apin_toluene': np.nan})


    #print('apin_toluene' in apin_toluene['parentspecies'].values)

    #apin_toluene_train, apin_toluene_test=train_test_split(apin_toluene, test_size=0.33, random_state=42)

    #print("apin_toluene_train: ", apin_toluene_train['parentspecies'].value_counts())
    #print("apin_toluene_test: ", apin_toluene_test['parentspecies'].value_counts())

    #print("a:",apin_toluene['parentspecies'].isna().sum())

    df_missing_y = apin_toluene[apin_toluene['parentspecies'].isna()]
    #print(df_missing_y.head())

    df_no_missing_y = apin_toluene.dropna(subset=['parentspecies'])

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

    #print(apin_toluene['parentspecies'].value_counts())
    # Replace the missing y values with the predicted values
    apin_toluene.loc[apin_toluene['parentspecies'].isna(), 'parentspecies'] = y_missing_pred
    #print(apin_toluene['parentspecies'].value_counts())

    for index, row in apin_toluene.iterrows():
        id_match = row['Id']
        new_value = row['parentspecies']
        
        train.loc[train['Id'] == id_match, 'parentspecies'] = new_value

    #print(train['parentspecies'].value_counts())
    return train