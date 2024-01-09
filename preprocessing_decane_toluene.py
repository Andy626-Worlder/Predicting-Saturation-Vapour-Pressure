#%%
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

def pre_decane_toluene(train):
    selected_categories_for_decane_toluene = ['decane', 'toluene', 'decane_toluene']
    decane_toluene = train[train['parentspecies'].isin(selected_categories_for_decane_toluene)]
    #print(decane_toluene.head())

    #print(decane_toluene['parentspecies'].value_counts())

    decane_toluene['parentspecies'] = decane_toluene['parentspecies'].replace({'decane_toluene': np.nan})


    #print('decane_toluene' in decane_toluene['parentspecies'].values)

    #decane_toluene_train, decane_toluene_test=train_test_split(decane_toluene, test_size=0.33, random_state=42)

    #print("decane_toluene_train: ", decane_toluene_train['parentspecies'].value_counts())
    #print("decane_toluene_test: ", decane_toluene_test['parentspecies'].value_counts())

    #print("a:",decane_toluene['parentspecies'].isna().sum())

    df_missing_y = decane_toluene[decane_toluene['parentspecies'].isna()]
    #print(df_missing_y.head())

    df_no_missing_y = decane_toluene.dropna(subset=['parentspecies'])

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

    #print(decane_toluene['parentspecies'].value_counts())
    # Replace the missing y values with the predicted values
    decane_toluene.loc[decane_toluene['parentspecies'].isna(), 'parentspecies'] = y_missing_pred
    #print(decane_toluene['parentspecies'].value_counts())

    for index, row in decane_toluene.iterrows():
        id_match = row['Id']
        new_value = row['parentspecies']
        
        train.loc[train['Id'] == id_match, 'parentspecies'] = new_value

    #print(train['parentspecies'].value_counts())
    return train