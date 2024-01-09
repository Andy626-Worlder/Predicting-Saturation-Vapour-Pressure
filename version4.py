#%%
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import GridSearchCV, RepeatedKFold, cross_val_score,cross_val_predict,KFold
from sklearn.metrics import make_scorer, r2_score, mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from preprocessing_apin_toluene import pre_apin_toluene
from preprocessing_apin_decane import pre_apin_decane
from preprocessing_decane_toluene import pre_decane_toluene
from preprocessing_apin_decane_toluene import pre_apin_decane_toluene

#%%
# load data
Train = pd.read_csv("train.csv",index_col='Id')
Test = pd.read_csv("test.csv",index_col='Id')
Train['pSat_Pa']=Train['pSat_Pa'].apply(np.log10)


#%%
observation_counts = Train['parentspecies'].value_counts(dropna=False)

print(observation_counts)



#%%
"""Train=pre_apin_decane(Train)
Train=pre_apin_toluene(Train)
Train=pre_decane_toluene(Train)
Train=pre_apin_decane_toluene(Train)


Test=pre_apin_decane(Test)
Test=pre_apin_toluene(Test)
#df_test=pre_decane_toluene(Test)
Test=pre_apin_decane_toluene(Test)

print(Train.head()) 
print(Test.head())"""
#%%

#Only feature 'parentspecies' is categorical, therefore I decide to observe the relationship between 'parentspecies' and 'log10(pSat_Pa)'. I found that different species have different response interval, therefore I think it is possible to train specific model for each species.

#1.  Test set has no 'decane_toluene' in parentspecies, therefore we need to discard observation with 'decane_toluene' in training set
#2.  For 'apin_toluene', a obvious gap shows up.
#3.  Suspicious anomalies existed
#%%
"""plt.scatter(Train['parentspecies'],Train['pSat_Pa'])
plt.show()
plt.close()"""


#%%
#Function that detects anomalies
def find_outliers(model,X,y,sigma=3):
  """
  Detect anomalies and returns index of all anomalies
  """
  model.fit(X,y)
  y_pred = pd.Series(model.predict(X),index=y.index)
    
  #calculate residuals
  resid = y - y_pred
  mean_resid = resid.mean()
  std_resid = resid.std()
  
  #calculate z statistic, define outliers to be where |z|>sigma
  z = (resid-mean_resid)/std_resid
  outliers = z[abs(z)>sigma].index
  
  # print and plot the results
  print('R2=',model.score(X,y))
  print("mse=",mean_squared_error(y,y_pred))
  print('---------------------------------------')
  print('mean of residuals:',mean_resid)
  print('std of residuals:',std_resid)
  print('---------------------------------------')
  print(len(outliers),'outliers:')
  print(outliers.tolist())
  plt.figure(figsize=(15,5))
  ax_131 = plt.subplot(1,3,1)
  plt.plot(y,y_pred,'.')
  plt.plot(y.loc[outliers],y_pred.loc[outliers],'ro')
  plt.legend(['Accepted','Outlier'])
  plt.xlabel('y')
  plt.ylabel('y_pred')
  plt.show()
  plt.close()
  ax_132=plt.subplot(1,3,2)
  plt.plot(y,y-y_pred,'.')
  plt.plot(y.loc[outliers],y.loc[outliers]-y_pred.loc[outliers],'ro')
  plt.legend(['Accepted','Outlier'])
  plt.xlabel('y')
  plt.ylabel('y - y_pred')
  plt.show()
  plt.close()
  ax_133=plt.subplot(1,3,3)
  z.plot.hist(bins=50,ax=ax_133)
  z.loc[outliers].plot.hist(color='r',bins=50,ax=ax_133)
  plt.legend(['Accepted','Outlier'])
  plt.xlabel('z')
  plt.show()
  plt.close()
  return outliers



#%%
#Functions that finds and show relevance & distribution between observations and responses
def find_relevance(train):
  colnm = train.columns.tolist()
  mcorr = train[colnm].corr(method="spearman")
  mask = np.zeros_like(mcorr, dtype=np.bool_)
  mask[np.triu_indices_from(mask)]=True
  cmap = sns.diverging_palette(220,10,as_cmap=True)
  g = sns.heatmap(mcorr, mask=mask, cmap=cmap, square=True, annot=True, fmt='0.2f')
  plt.show()
  plt.close()
  corr_matrix = mcorr['pSat_Pa'].abs()
  print(corr_matrix)


#%%

def show_distribution(train,test):
  for column in test.columns:
    g = sns.kdeplot(train[column],color="Red",shade=True)
    g = sns.kdeplot(test[column],ax=g,color="Blue",shade=True)
    plt.xlabel(column)
    plt.ylabel("Frequency")
    g = g.legend(["train","test"])
    plt.show()
    plt.close()


#%%
#Function that normalize predictors of train / test

def scale_minmax(train,test):
  def scale(col):
    return (col-col.min())/(col.max()-col.min())
  scale_cols = [col for col in test.columns]
  train[scale_cols] = train[scale_cols].apply(scale,axis=0)
  test[scale_cols] = test[scale_cols].apply(scale,axis=0)
  return [train,test]

def scale_z_score(train,test):
  def scale(col):
    return (col-np.mean(col))/np.std(col)
  scale_cols = [col for col in test.columns]
  train[scale_cols] = train[scale_cols].apply(scale,axis=0)
  test[scale_cols] = test[scale_cols].apply(scale,axis=0)
  return [train,test]

def boxcox(train,test):
  col_transform = test.columns
  for col in ['MW','NumOfAtoms','NumOfC','NumOfO','NumOfN','NumHBondDonors','NumOfConf','NumOfConfUsed']:
    train.loc[:,col],_ = stats.boxcox(train.loc[:,col]+1)
    test.loc[:,col],_ = stats.boxcox(test.loc[:,col]+1)
  return [train,test]


#%%
#Make a R2 Scorer
r2_scorer = make_scorer(r2_score,greater_is_better=True)

#%%
#Functions that analyze maximum number of components using PCA
def analyze_all_features(X):
  """
  Plot proportion of variance explained & eigenvalues of each components
  """
  pca = PCA(n_components=len(X.columns)-1)
  pca.fit_transform(X)
  prop_var = pca.explained_variance_ratio_
  eigenvalues = pca.explained_variance_
  PC_numbers = np.arange(pca.n_components_) + 1
  fig, ax = plt.subplots(nrows=1, ncols=2)
  ax[0].plot(PC_numbers, 
         prop_var, 
         'ro-')
  ax[0].set_title('Overall proportrion explained', fontsize=8)
  ax[0].set_ylabel('Proportion of Variance', fontsize=8)
  ax[1].scatter(PC_numbers,eigenvalues)
  ax[1].set_title('Overall eigenvalues', fontsize=8)
  print(eigenvalues)
  plt.show()
  plt.close()

#%%
#### First, we analyze toluene species.
import warnings
# To suppress all future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def toluene_predict(train,test):
  train = train[train['parentspecies']=='toluene'].drop('parentspecies',axis=1)
  test = test[test['parentspecies']=='toluene'].drop('parentspecies',axis=1)
  
  # Then scale the observations
  train,test = scale_minmax(train,test) # 22 columns
  train,test = boxcox(train,test)
  train,test = scale_z_score(train,test)
  
  #outliers = find_outliers(RandomForestRegressor(),train.drop('pSat_Pa',axis=1),train['pSat_Pa'],sigma=3)
  # for better performance, I just write down the outliers detected from previous run
  #train = train.drop(outliers)
  # Then I find the relevance
  """find_relevance(train)
  show_distribution(train,test)"""
  # I observe that for feature ['peroxide'], there is significant difference between observations and testing set, therefore I should discard feature 'peroxide'.
  # I observe that feature ['aromatic.hydroxyl'] has constant value 0, therefore I should discard this one since is does not contribute to learning 
  train = train.drop(['peroxide','aromatic.hydroxyl'],axis=1)
  test = test.drop(['peroxide','aromatic.hydroxyl'],axis=1)
  
  X = train.drop('pSat_Pa',axis=1)
  y = train['pSat_Pa']
  
  # Then use PCA to analysis maximum number of components, I want to keep 95% of the variance explained
  analyze_all_features(X) #approximately 17 components
  def pca_select(train, test):
    pca = PCA(n_components=17)
    PC = pca.fit_transform(train)
    pca_train = pd.DataFrame(data = PC, columns = ['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10','PC11','PC12','PC13','PC14','PC15','PC16','PC17'],index=train.index)
    PC_test = pca.transform(test)
    pca_test = pd.DataFrame(data = PC_test, columns = ['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10','PC11','PC12','PC13','PC14','PC15','PC16','PC17'],index=test.index)
    return [pca_train,pca_test]
  X,test = pca_select(X,test)
  
  # Then we select andd train model. 
  def train_model(model, param_grid=[], X=[], y=[], splits=5, repeats=2):
    # create cross-validation method
    rkfold = RepeatedKFold(n_splits=splits, n_repeats=repeats)
    # setup grid search parameters
    gsearch = GridSearchCV(model, param_grid, cv=rkfold,
                           scoring=r2_scorer,
                           verbose=2, return_train_score=True)
    gsearch.fit(X,y)
    # extract best model from the grid
    model = gsearch.best_estimator_        
    best_idx = gsearch.best_index_
    # evaluate result
    grid_results = pd.DataFrame(gsearch.cv_results_)
    cv_mean = abs(grid_results.loc[best_idx,'mean_test_score'])
    cv_std = grid_results.loc[best_idx,'std_test_score']
    # combine mean and std cv-score in to a pandas series
    cv_score = pd.Series({'mean':cv_mean,'std':cv_std})
    # predict y using the fitted model
    y_pred = model.predict(X)
    y_prediction = pd.DataFrame(data=model.predict(test),index=test.index,columns=['target'])
    # print stats on model performance         
    print('----------------------')
    print(model)
    print('score=',model.score(X,y))
    print('cross_val: mean=',cv_mean,', std=',cv_std)
    return y_prediction
  KN = KNeighborsRegressor()
  param_gridkn = {'n_neighbors':np.arange(3,13,1)}
  svr = SVR(kernel='poly')
  param_grid1 = {
    'degree': [1]}
  RF = RandomForestRegressor()
  param_grid = {'n_estimators':[550], #550 > 500, 550>600
                'max_features':[9], #9>8 > 7
                'min_samples_split':[11],# 11 > 12, 11>10
                'max_depth':[21],#21 > 22, 21 > 20
                'min_samples_leaf': [2]}
                
  param_grid2 = {'n_estimators':[100,150,200],
              'max_features':[2,3,4,5],
              'min_samples_split':[2,4,6]}
  y_prediction = train_model(RF,param_grid=param_grid,X=X,y=y,splits=10, repeats=1)
  print(y_prediction.describe())
  return y_prediction
  
t_pred = toluene_predict(Train,Test)
#print(toluene_pred.describe())

#%%      
R2= 0.9595307221647719
mse= 0.21069702940131924
"""mean of residuals: 0.0022727148430369427
std of residuals: 0.4590243810741256"""
"""---------------------------------------
261 outliers:
[150, 173, 1314, 1565, 1702, 1987, 2089, 2184, 2964, 2999, 3862, 3960, 4576, 5327, 5701, 5722, 6644, 8748, 10353, 10376, 12436, 12690, 12876, 12881, 14361, 14428, 14942, 15184, 15323, 15642, 16063, 16404, 16446, 17503, 17576, 18981, 19467, 20085, 20641, 20653, 21443, 21980, 22013, 22403, 22533, 23018, 23172, 23195, 23293, 23316, 24086, 26281, 26849, 26909, 27248, 27419, 27567, 28072, 28771, 29538, 29684, 29803, 29853, 29873, 30547, 30954, 31002, 31117, 32650, 32671, 32986, 33034, 33289, 33311, 33516, 33670, 34550, 35466, 35524, 35706, 35793, 35886, 36082, 36109, 36117, 36159, 36184, 36431, 37717, 37906, 38293, 40561, 40592, 40929, 41123, 41319, 42740, 44523, 45535, 46227, 47404, 47493, 47566, 47943, 49402, 49705, 49773, 50059, 51516, 51662, 51829, 51899, 51911, 54141, 55299, 55347, 55351, 56291, 56433, 56787, 56977, 56991, 57098, 58074, 59063, 59978, 60111, 60779, 60925, 61321, 63315, 63460, 68198, 71550, 73071, 75102, 76291, 76805, 77200, 78688, 78949, 79338, 79379, 80350, 81208, 81750, 82036, 83037, 83273, 83306, 83332, 83381, 83387, 83388, 83403, 83415, 83466, 83468, 83477, 83676, 83701, 83702, 83744, 83757, 83784, 83786, 83787, 83797, 
"""


#%%
#Next we work on None species
def None_predict(train,test):
  train = train[train['parentspecies']=='None'].drop('parentspecies',axis=1)
  test = test[test['parentspecies']=='None'].drop('parentspecies',axis=1)
  
  #outliers = find_outliers(RandomForestRegressor(),train.drop('pSat_Pa',axis=1),train['pSat_Pa'],sigma=3)
  #train = train.drop(outliers)
  # Then I find the relevance and observe distribution
  #find_relevance(train)
  #show_distribution(train,test)
  # features with all-0: ['C.C.C.O.in.non.aromatic.ring','aromatic.hydroxyl']
  # features with non-uniform distribution: ['C.C..non.aromatic.','NumOfC']
  # features with suspicious non-uniform distribution: ['NumOfN','NumHBondDonors']
  train = train.drop(['C.C.C.O.in.non.aromatic.ring','aromatic.hydroxyl','C.C..non.aromatic.'],axis=1)
  test = test.drop(['C.C.C.O.in.non.aromatic.ring','aromatic.hydroxyl','C.C..non.aromatic.'],axis=1)
  
  # Then scale the observations
  #train,test = scale_minmax(train,test) # 22 columns
  train,test = boxcox(train,test)
  train,test = scale_z_score(train,test)
  X = train.drop('pSat_Pa',axis=1)
  y = train['pSat_Pa']
  
  # Then use PCA to analysis maximum number of components, I want to keep 95% of the variance explained
  #analyze_all_features(X) #approximately 13 components
  def pca_select(train, test):
    pca = PCA(n_components=16)
    PC = pca.fit_transform(train)
    pca_train = pd.DataFrame(data = PC, columns = ['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10','PC11','PC12','PC13','PC14','PC15','PC16'],index=train.index)
    PC_test = pca.transform(test)
    pca_test = pd.DataFrame(data = PC_test, columns = ['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10','PC11','PC12','PC13','PC14','PC15','PC16'],index=test.index)
    return [pca_train,pca_test]
  X,test = pca_select(X,test)
  
  # Then we select andd train model. 
  def train_model(model, param_grid=[], X=[], y=[], splits=5, repeats=2):
    # create cross-validation method
    rkfold = RepeatedKFold(n_splits=splits, n_repeats=repeats)
    # setup grid search parameters
    gsearch = GridSearchCV(model, param_grid, cv=rkfold,
                           scoring=r2_scorer,
                           verbose=2, return_train_score=True)
    gsearch.fit(X,y)
    # extract best model from the grid
    model = gsearch.best_estimator_        
    best_idx = gsearch.best_index_
    # evaluate result
    grid_results = pd.DataFrame(gsearch.cv_results_)
    cv_mean = abs(grid_results.loc[best_idx,'mean_test_score'])
    cv_std = grid_results.loc[best_idx,'std_test_score']
    # combine mean and std cv-score in to a pandas series
    cv_score = pd.Series({'mean':cv_mean,'std':cv_std})
    # predict y using the fitted model
    y_pred = model.predict(X)
    y_prediction = pd.DataFrame(data=model.predict(test),index=test.index,columns=['target'])
    # print stats on model performance         
    print('----------------------')
    print(model)
    print('score=',model.score(X,y))
    print('cross_val: mean=',cv_mean,', std=',cv_std)
    return y_prediction
  svr = SVR(kernel='poly')
  param_grid1 = {
    'degree': [1]}
  RF = RandomForestRegressor()
  param_grid = {'n_estimators':[575], # 575>600 > 550 & 650
                'max_features':[12], #12>10>8, 12>14
                'min_samples_split':[7], #6>5,6>8
                'max_depth':[27], #27>28,29>31,27>26,25
                'min_samples_leaf': [2]} #2>1
  y_prediction = train_model(RF,param_grid=param_grid,X=X,y=y,splits=10, repeats=1)
  print(y_prediction.describe())
  return y_prediction
  
N_pred = None_predict(Train,Test)

#%%
#Now we work on apin

def apin_predict(train,test):
  train = train[train['parentspecies']=='apin'].drop('parentspecies',axis=1)
  test = test[test['parentspecies']=='apin'].drop('parentspecies',axis=1)
  
  outliers = find_outliers(RandomForestRegressor(),train.drop('pSat_Pa',axis=1),train['pSat_Pa'],sigma=3)
  # for better performance, I just write down the outliers detected from previous run
  #train = train.drop(outliers)
  
  # Then I find the relevance and observe distribution
  #find_relevance(train)
  #show_distribution(train,test)
  # features with all-0: ['C.C.C.O.in.non.aromatic.ring','ester','ether..alicyclic.','C.C.C.O.in.non.aromatic.ring','aromatic.hydroxyl','nitroester']
  # features with non-uniform distribution: []
  # features with suspicious non-uniform distribution: []
  train = train.drop(['C.C.C.O.in.non.aromatic.ring','ester','ether..alicyclic.','C.C.C.O.in.non.aromatic.ring','nitro','aromatic.hydroxyl','nitroester'],axis=1)
  test = test.drop(['C.C.C.O.in.non.aromatic.ring','ester','ether..alicyclic.','C.C.C.O.in.non.aromatic.ring','nitro','aromatic.hydroxyl','nitroester'],axis=1)
  
  # Then scale the observations
  #train,test = scale_minmax(train,test)
  train,test = boxcox(train,test)
  train,test = scale_z_score(train,test)
  
  X = train.drop('pSat_Pa',axis=1)
  y = train['pSat_Pa']
  
  # Then use PCA to analysis maximum number of components, I want to keep 95% of the variance explained
  analyze_all_features(X) #approximately 13 components
  def pca_select(train, test):
    pca = PCA(n_components=13)
    PC = pca.fit_transform(train)
    pca_train = pd.DataFrame(data = PC, columns = ['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10','PC11','PC12','PC13'],index=train.index)
    PC_test = pca.transform(test)
    pca_test = pd.DataFrame(data = PC_test, columns = ['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10','PC11','PC12','PC13'],index=test.index)
    return [pca_train,pca_test]
  X,test = pca_select(X,test)
  
  # Then we select andd train model. 
  def train_model(model, param_grid=[], X=[], y=[], splits=5, repeats=2):
    # create cross-validation method
    rkfold = RepeatedKFold(n_splits=splits, n_repeats=repeats)
    # setup grid search parameters
    gsearch = GridSearchCV(model, param_grid, cv=rkfold,
                           scoring=r2_scorer,
                           verbose=2, return_train_score=True)
    gsearch.fit(X,y)
    # extract best model from the grid
    model = gsearch.best_estimator_        
    best_idx = gsearch.best_index_
    # evaluate result
    grid_results = pd.DataFrame(gsearch.cv_results_)
    cv_mean = abs(grid_results.loc[best_idx,'mean_test_score'])
    cv_std = grid_results.loc[best_idx,'std_test_score']
    # combine mean and std cv-score in to a pandas series
    cv_score = pd.Series({'mean':cv_mean,'std':cv_std})
    # predict y using the fitted model
    y_pred = model.predict(X)
    y_prediction = pd.DataFrame(data=model.predict(test),index=test.index,columns=['target'])
    # print stats on model performance         
    print('----------------------')
    print(model)
    print('score=',model.score(X,y))
    print('cross_val: mean=',cv_mean,', std=',cv_std)
    return y_prediction
  GB = GradientBoostingRegressor()
  param_gridgb = {'n_estimators':[200], #200>250>150
              'max_depth':[2],#2>1
              'min_samples_split':[7]}#7>6>8
  svr = SVR(kernel='poly')
  param_grid1 = {
    'degree': [1]}
  RF = RandomForestRegressor()
  param_grid = {'n_estimators':[650], #500,525,550 < 600<650
                'max_features':[6], #1,2,3,4<6, 6>8
                'min_samples_split':[14],#12<14, 15<14
                'max_depth':[25],#29>33, 29>17, 27>29, 25>27
                'min_samples_leaf': [2]} #2>1
  y_prediction = train_model(svr,param_grid=param_grid1,X=X,y=y,splits=10, repeats=1)
  print(y_prediction.describe())
  return y_prediction
  
a_pred = apin_predict(Train,Test)
#print(apin_pred.describe())


# ActivisionFunction
#%%
#Then comes decane
def decane_predict(train,test):
  train = train[train['parentspecies']=='decane'].drop('parentspecies',axis=1)
  test = test[test['parentspecies']=='decane'].drop('parentspecies',axis=1)
  
  outliers = find_outliers(RandomForestRegressor(),train.drop('pSat_Pa',axis=1),train['pSat_Pa'],sigma=3)
  # for better performance, I just write down the outliers detected from previous run
  #train = train.drop(outliers)
  
  # Then I find the relevance and observe distribution
  #find_relevance(train)
  #show_distribution(train,test)
  # features with all-0: ['C.C.C.O.in.non.aromatic.ring','ester','ether..alicyclic.','C.C.C.O.in.non.aromatic.ring','nitro','aromatic.hydroxyl','peroxide','nitroester']
  # features with non-uniform distribution: []
  # features with suspicious non-uniform distribution: []
  train = train.drop(['C.C.C.O.in.non.aromatic.ring','ester','ether..alicyclic.','C.C.C.O.in.non.aromatic.ring','nitro','aromatic.hydroxyl','peroxide','nitroester'],axis=1)
  test = test.drop(['C.C.C.O.in.non.aromatic.ring','ester','ether..alicyclic.','C.C.C.O.in.non.aromatic.ring','nitro','aromatic.hydroxyl','peroxide','nitroester'],axis=1)
  
  # Then scale the observations
  #train,test = scale_minmax(train,test)
  train,test = boxcox(train,test)
  train,test = scale_z_score(train,test)
  
  X = train.drop('pSat_Pa',axis=1)
  y = train['pSat_Pa']
  
  # Then use PCA to analysis maximum number of components, I want to keep 95% of the variance explained
  #analyze_all_features(X) #approximately 12 components
  def pca_select(train, test):
    pca = PCA(n_components=12)
    PC = pca.fit_transform(train)
    pca_train = pd.DataFrame(data = PC, columns = ['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10','PC11','PC12'],index=train.index)
    PC_test = pca.transform(test)
    pca_test = pd.DataFrame(data = PC_test, columns = ['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10','PC11','PC12'],index=test.index)
    return [pca_train,pca_test]
  X,test = pca_select(X,test)
  
  # Then we select andd train model. 
  def train_model(model, param_grid=[], X=[], y=[], splits=5, repeats=2):
    # create cross-validation method
    rkfold = RepeatedKFold(n_splits=splits, n_repeats=repeats)
    # setup grid search parameters
    gsearch = GridSearchCV(model, param_grid, cv=rkfold,
                           scoring=r2_scorer,
                           verbose=2, return_train_score=True)
    gsearch.fit(X,y)
    # extract best model from the grid
    model = gsearch.best_estimator_        
    best_idx = gsearch.best_index_
    # evaluate result
    grid_results = pd.DataFrame(gsearch.cv_results_)
    cv_mean = abs(grid_results.loc[best_idx,'mean_test_score'])
    cv_std = grid_results.loc[best_idx,'std_test_score']
    # combine mean and std cv-score in to a pandas series
    cv_score = pd.Series({'mean':cv_mean,'std':cv_std})
    # predict y using the fitted model
    y_pred = model.predict(X)
    y_prediction = pd.DataFrame(data=model.predict(test),index=test.index,columns=['target'])
    # print stats on model performance         
    print('----------------------')
    print(model)
    print('score=',model.score(X,y))
    print('cross_val: mean=',cv_mean,', std=',cv_std)
    return y_prediction
  GB = GradientBoostingRegressor()
  param_gridgb = {'n_estimators':[250], #250 > 150, 350
              'max_depth':[2],#2>3&1
              'min_samples_split':[5]}#5>6
  KN = KNeighborsRegressor()
  param_gridkn = {'n_neighbors':np.arange(3,13,1)}
  svr = SVR(kernel='poly')
  param_grid1 = {
    'degree': [1,2,3,4,5,6,7]}
  RF = RandomForestRegressor()
  param_grid = {'n_estimators':[650], #500,525,550 < 600
                'max_features':[6], #1,2,3 4,5<6
                'min_samples_split':[13],#11&15<13
                'max_depth':[25],#27>29&31
                'min_samples_leaf': [2]} #2>1
  y_prediction = train_model(svr,param_grid=param_grid1,X=X,y=y,splits=10, repeats=1)
  print(y_prediction.describe())
  return y_prediction
  
d_pred = decane_predict(Train,Test)
#print(decane_pred.describe())

#%%
#Now apin_toluene
def at_predict(train,test):
  train = train[train['parentspecies']=='apin_toluene'].drop('parentspecies',axis=1)
  test = test[test['parentspecies']=='apin_toluene'].drop('parentspecies',axis=1)
  
  outliers = find_outliers(RandomForestRegressor(),train.drop('pSat_Pa',axis=1),train['pSat_Pa'],sigma=3)
  # for better performance, I just write down the outliers detected from previous run
  #train = train.drop(outliers)
  
  # Then I find the relevance and observe distribution
  #find_relevance(train)
  #show_distribution(train,test)
  # features with all-0: ['C.C.C.O.in.non.aromatic.ring','ester','ether..alicyclic.','nitro','aromatic.hydroxyl','peroxide','nitroester']
  # features with non-uniform distribution: ['aldehyde']
  # features with suspicious non-uniform distribution: []
  train = train.drop(['C.C.C.O.in.non.aromatic.ring','ester','ether..alicyclic.','nitro','aromatic.hydroxyl','peroxide','nitroester','aldehyde','C.C..non.aromatic.', 'carbonylperoxynitrate'],axis=1)
  test = test.drop(['C.C.C.O.in.non.aromatic.ring','ester','ether..alicyclic.','nitro','aromatic.hydroxyl','peroxide','nitroester','aldehyde','C.C..non.aromatic.', 'carbonylperoxynitrate'],axis=1)
  
  # Then scale the observations
  #train,test = scale_minmax(train,test) # 22 columns
  train,test = boxcox(train,test)
  train,test = scale_z_score(train,test)
  
  X = train.drop('pSat_Pa',axis=1)
  y = train['pSat_Pa']

  # Then use PCA to analysis maximum number of components, I want to keep 95% of the variance explained
  #analyze_all_features(X) #approximately 9 components
  def pca_select(train, test):
    pca = PCA(n_components=10)
    PC = pca.fit_transform(train)
    pca_train = pd.DataFrame(data = PC, columns = ['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10'],index=train.index)
    PC_test = pca.transform(test)
    pca_test = pd.DataFrame(data = PC_test, columns = ['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10'],index=test.index)
    return [pca_train,pca_test]
  X,test = pca_select(X,test)
  
  # Then we select andd train model. 
  def train_model(model, param_grid=[], X=[], y=[], splits=5, repeats=2):
    # create cross-validation method
    rkfold = RepeatedKFold(n_splits=splits, n_repeats=repeats)
    # setup grid search parameters
    gsearch = GridSearchCV(model, param_grid, cv=rkfold,
                           scoring=r2_scorer,
                           verbose=2, return_train_score=True)
    gsearch.fit(X,y)
    # extract best model from the grid
    model = gsearch.best_estimator_        
    best_idx = gsearch.best_index_
    # evaluate result
    grid_results = pd.DataFrame(gsearch.cv_results_)
    cv_mean = abs(grid_results.loc[best_idx,'mean_test_score'])
    cv_std = grid_results.loc[best_idx,'std_test_score']
    # combine mean and std cv-score in to a pandas series
    cv_score = pd.Series({'mean':cv_mean,'std':cv_std})
    # predict y using the fitted model
    y_pred = model.predict(X)
    y_prediction = pd.DataFrame(data=model.predict(test),index=test.index,columns=['target'])
    # print stats on model performance         
    print('----------------------')
    print(model)
    print('score=',model.score(X,y))
    print('cross_val: mean=',cv_mean,', std=',cv_std)
    return y_prediction
  svr = SVR(kernel='poly')
  param_grid1 = {
    'degree': [1,2,3,4,5,6,7]}
  RF = RandomForestRegressor()
  param_grid = {'n_estimators':[620], #620>600>550>500&450
                'max_features':[8], #8>7>5&9
                'min_samples_split':[5],#5>4&6>7>9&11
                'max_depth':[23],#23>24>22&26
                'min_samples_leaf': [2]}#2>1&3
  y_prediction = train_model(svr,param_grid=param_grid1,X=X,y=y,splits=10, repeats=1)
  print(y_prediction.describe())
  return y_prediction
  
at_pred = at_predict(Train,Test)
#print(at_pred.describe())

#%%
#Then apin_decane
def ad_predict(train,test):
  from sklearn.svm import LinearSVR
  train = train[train['parentspecies']=='apin_decane'].drop('parentspecies',axis=1)
  test = test[test['parentspecies']=='apin_decane'].drop('parentspecies',axis=1)
  
  outliers = find_outliers(RandomForestRegressor(),train.drop('pSat_Pa',axis=1),train['pSat_Pa'],sigma=3)
  # for better performance, I just write down the outliers detected from previous run
  #outliers = []
  #train = train.drop(outliers)
  
  # Then I find the relevance and observe distribution
  #find_relevance(train)
  #show_distribution(train,test)
  # features with all-0: ['C.C.C.O.in.non.aromatic.ring','ester','ether..alicyclic.','nitro','aromatic.hydroxyl','peroxide','nitroester']
  # features with non-uniform distribution: ['aldehyde']
  # features with suspicious non-uniform distribution: []
  train = train.drop(['C.C.C.O.in.non.aromatic.ring','ester','ether..alicyclic.','nitro','aromatic.hydroxyl','peroxide','nitroester','aldehyde','C.C..non.aromatic.', 'carbonylperoxynitrate','carbonylperoxyacid'],axis=1)
  test = test.drop(['C.C.C.O.in.non.aromatic.ring','ester','ether..alicyclic.','nitro','aromatic.hydroxyl','peroxide','nitroester','aldehyde','C.C..non.aromatic.', 'carbonylperoxynitrate','carbonylperoxyacid'],axis=1)
  
  # Then scale the observations
  # train,test = scale_minmax(train,test) # 22 columns
  train,test = boxcox(train,test)
  train,test = scale_z_score(train,test)
  
  X = train.drop('pSat_Pa',axis=1)
  y = train['pSat_Pa']
  
  # Then use PCA to analysis maximum number of components, I want to keep 95% of the variance explained
  #analyze_all_features(X) #approximately 8 components
  def pca_select(train, test):
    pca = PCA(n_components=9)
    PC = pca.fit_transform(train)
    pca_train = pd.DataFrame(data = PC, columns = ['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9'],index=train.index)
    PC_test = pca.transform(test)
    pca_test = pd.DataFrame(data = PC_test, columns = ['PC1','PC2','PC3','PC4','PC5','PC6','PC7','PC8','PC9'],index=test.index)
    return [pca_train,pca_test]
  X,test = pca_select(X,test)
  
  # Then we select andd train model. 
  def train_model(model, param_grid=[], X=[], y=[], splits=5, repeats=2):
    # create cross-validation method
    rkfold = RepeatedKFold(n_splits=splits, n_repeats=repeats)
    # setup grid search parameters
    gsearch = GridSearchCV(model, param_grid, cv=rkfold,
                           scoring=r2_scorer,
                           verbose=2, return_train_score=True)
    gsearch.fit(X,y)
    # extract best model from the grid
    model = gsearch.best_estimator_        
    best_idx = gsearch.best_index_
    # evaluate result
    grid_results = pd.DataFrame(gsearch.cv_results_)
    cv_mean = abs(grid_results.loc[best_idx,'mean_test_score'])
    cv_std = grid_results.loc[best_idx,'std_test_score']
    # combine mean and std cv-score in to a pandas series
    cv_score = pd.Series({'mean':cv_mean,'std':cv_std})
    # predict y using the fitted model
    y_pred = model.predict(X)
    y_prediction = pd.DataFrame(data=model.predict(test),index=test.index,columns=['target'])
    # print stats on model performance         
    print('----------------------')
    print(model)
    print('score=',model.score(X,y))
    print('cross_val: mean=',cv_mean,', std=',cv_std)
    return y_prediction
  svr = LinearSVR()
  param_grid1 = {
    'C': [0.2],  # Regularization parameter
    'max_iter': [2200],        # Maximum number of iterations
    'tol': [0.02]              # Tolerance for stopping criteria
  }
  RF = RandomForestRegressor()
  param_grid = {'n_estimators':[125], #125>150>100&200&250
                'max_features':[15], #15>14>10&12&16
                'min_samples_split':[6], #6>5&4&7&8
                'max_depth':[8],#8>11>12,13,14>16>18&20
                'min_samples_leaf': [3]} #3>2&1
  y_prediction = train_model(svr,param_grid=param_grid1,X=X,y=y,splits=10, repeats=2)
  print(y_prediction.describe())
  return y_prediction
  
ad_pred = ad_predict(Train,Test)
#print(ad_pred.describe())

#%%
#last is apin_decane_toluene
def adt_predict(train,test):
  train = train[train['parentspecies']=='apin_decane_toluene'].drop('parentspecies',axis=1)
  test = test[test['parentspecies']=='apin_decane_toluene'].drop('parentspecies',axis=1)
  
  #outliers = find_outliers(RandomForestRegressor(),train.drop('pSat_Pa',axis=1),train['pSat_Pa'],sigma=3)
  # for better performance, I just write down the outliers detected from previous run
  #outliers = []
  #train = train.drop(outliers)
  
  # Then I find the relevance and observe distribution
  #find_relevance(train)
  #show_distribution(train,test)
  # features with all-0: ['C.C..non.aromatic.','C.C.C.O.in.non.aromatic.ring','ester','ether..alicyclic.','nitro','aromatic.hydroxyl','peroxide','carbonylperoxyacid','nitroester']
  # features with non-uniform distribution: ['aldehyde']
  # features with suspicious non-uniform distribution: []
  train = train.drop(['C.C..non.aromatic.','C.C.C.O.in.non.aromatic.ring','ester','ether..alicyclic.','nitro','aromatic.hydroxyl','aldehyde','peroxide','carbonylperoxyacid','nitroester','NumOfO', 'NumOfN', 'NumHBondDonors', 'nitrate', 'carbonylperoxynitrate', 'hydroperoxide'],axis=1)
  test = test.drop(['C.C..non.aromatic.','C.C.C.O.in.non.aromatic.ring','ester','ether..alicyclic.','nitro','aromatic.hydroxyl','aldehyde','peroxide','carbonylperoxyacid','nitroester','NumOfO', 'NumOfN', 'NumHBondDonors', 'nitrate', 'carbonylperoxynitrate', 'hydroperoxide'],axis=1)
  
  # Then scale the observations
  #train,test = scale_minmax(train,test) # 22 columns
  #train,test = boxcox(train,test)
  train,test = scale_z_score(train,test)
  
  X = train.drop('pSat_Pa',axis=1)
  y = train['pSat_Pa']
  
  # Very little observations, deploy inflexible method
  # We select andd train model. 
  def train_model(model, param_grid=[], X=[], y=[], splits=5, repeats=2):
    # create cross-validation method
    rkfold = RepeatedKFold(n_splits=splits, n_repeats=repeats)
    # setup grid search parameters
    gsearch = GridSearchCV(model, param_grid, cv=rkfold,
                           scoring=r2_scorer,
                           verbose=2, return_train_score=True)
    gsearch.fit(X,y)
    # extract best model from the grid
    model = gsearch.best_estimator_        
    best_idx = gsearch.best_index_
    # evaluate result
    grid_results = pd.DataFrame(gsearch.cv_results_)
    cv_mean = abs(grid_results.loc[best_idx,'mean_test_score'])
    cv_std = grid_results.loc[best_idx,'std_test_score']
    # combine mean and std cv-score in to a pandas series
    cv_score = pd.Series({'mean':cv_mean,'std':cv_std})
    # predict y using the fitted model
    y_pred = model.predict(X)
    y_prediction = pd.DataFrame(data=model.predict(test),index=test.index,columns=['target'])
    # print stats on model performance         
    print('----------------------')
    print(model)
    print('score=',model.score(X,y))
    print('cross_val: mean=',cv_mean,', std=',cv_std)
    return y_prediction
  svr = SVR(kernel='poly')
  param_grid = {
    'degree': [1,2,3,4,5,6,7]}
  y_prediction = train_model(svr,param_grid=param_grid,X=X,y=y,splits=5, repeats=2)
  print(y_prediction.describe())
  return y_prediction
  
adt_pred = adt_predict(Train,Test)
#print(ad_pred.describe())

#%%
#Predict all, compensate for loss caused by classifying according to species
def all_predict():
  train = pd.get_dummies(pd.read_csv("train.csv",index_col="Id"),columns=['parentspecies'])
  test = pd.get_dummies(pd.read_csv("test.csv",index_col="Id"),columns=['parentspecies'])
  # complement for Test, since Test has no decane_toluene
  new_column = pd.Series(0,index=test.index,name='parentspecies_decane_toluene')
  test.insert(30,'parentspecies_decane_toluene',new_column)
  
  # Then scale the observations
  # train,test = scale_minmax(train,test) # 22 columns
  train,test = boxcox(train,test)
  train,test = scale_z_score(train,test)
  test = test.drop('parentspecies_decane_toluene',axis=1)
  train = train.drop('parentspecies_decane_toluene',axis=1)
  
  X = train.drop('pSat_Pa',axis=1)
  y = train['pSat_Pa']
  
  # Then use PCA to analysis maximum number of components, I want to keep 95% of the variance explained
  analyze_all_features(X) #approximately 8 components
  def pca_select(train, test):
    pca = PCA(n_components=23)
    PC = pca.fit_transform(train)
    pca_train = pd.DataFrame(data = PC,index=train.index)
    PC_test = pca.transform(test)
    pca_test = pd.DataFrame(data = PC_test,index=test.index)
    return [pca_train,pca_test]
  X,test = pca_select(X,test)
  
  def train_model(model, param_grid=[], X=[], y=[], splits=5, repeats=2):
    # create cross-validation method
    rkfold = RepeatedKFold(n_splits=splits, n_repeats=repeats)
    # setup grid search parameters
    gsearch = GridSearchCV(model, param_grid, cv=rkfold,
                           scoring=r2_scorer,
                           verbose=2, return_train_score=True)
    gsearch.fit(X,y)
    # extract best model from the grid
    model = gsearch.best_estimator_        
    best_idx = gsearch.best_index_
    # evaluate result
    grid_results = pd.DataFrame(gsearch.cv_results_)
    cv_mean = abs(grid_results.loc[best_idx,'mean_test_score'])
    cv_std = grid_results.loc[best_idx,'std_test_score']
    # combine mean and std cv-score in to a pandas series
    cv_score = pd.Series({'mean':cv_mean,'std':cv_std})
    # predict y using the fitted model
    y_pred = model.predict(X)
    y_prediction = pd.DataFrame(data=model.predict(test),index=test.index,columns=['target'])
    # print stats on model performance         
    print('----------------------')
    print(model)
    print('score=',model.score(X,y))
    print('cross_val: mean=',cv_mean,', std=',cv_std)
    return y_prediction
  RF = RandomForestRegressor()
  param_grid = {'n_estimators':[600,650], #550 > 500, 550>600
                'max_features':[12], #9>8 > 7
                'min_samples_split':[11],# 11 > 12, 11>10
                'max_depth':[25],#21 > 22, 21 > 20
                'min_samples_leaf': [1]}
  y_prediction = train_model(RF,param_grid=param_grid,X=X,y=y,splits=6, repeats=1)
  print(y_prediction.describe())
  return y_prediction

all_pred = all_predict()



#%%
#
#Sumamry
result = pd.concat([t_pred,a_pred,d_pred,at_pred,ad_pred,N_pred,adt_pred],axis=0)
result = result.sort_index()
print(result.describe())
print(Train[['pSat_Pa']].describe())
result.to_csv("version5.csv",index=True)
# %%

