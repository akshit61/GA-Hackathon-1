# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 14:52:44 2020

@author: Akshit
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as sc
import statsmodels.api as sm
import time
from itertools import compress
import warnings
warnings.filterwarnings('ignore')
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder,PowerTransformer
from sklearn.model_selection import train_test_split, GridSearchCV,RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier,ExtraTreesClassifier,GradientBoostingClassifier,VotingClassifier
from sklearn import metrics
from sklearn.feature_selection import SelectKBest,RFE,RFECV,f_classif
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
pd.set_option('display.max_columns',None)




def num_features(dataset):
    '''
    Extract numerical features from dataset

    Parameters
    ----------
    dataset : Dataframe
        DESCRIPTION.

    Returns
    -------
    numeric_data : Dataframe
        DESCRIPTION.

    '''
    numeric_data = dataset.select_dtypes(exclude = 'object')
    return numeric_data

def cat_features(dataset):
    '''
    Extract categorical features from dataset

    Parameters
    ----------
    dataset : Dataframe
        DESCRIPTION.

    Returns
    -------
    categorical_data : Dataframe
        DESCRIPTION.

    '''
    categorical_data = dataset.select_dtypes(include = 'object')
    return categorical_data

def check_dtypes(dataset):
    '''
    returns datatype for each feature in the dataframe

    Parameters
    ----------
    dataset : Dataframe
        DESCRIPTION.

    Returns
    -------
    Series
        DESCRIPTION.

    '''
    return dataset.dtypes

def missing_data(dataset):
    '''
    Return total missing data and percentage of missing data for each feature

    Parameters
    ----------
    dataset : Dataframe
        DESCRIPTION.

    Returns
    -------
    missing_data : Dataframe
        DESCRIPTION.

    '''
    total_missing = dataset.isnull().sum()
    percent_misiing = (dataset.isnull().sum()/dataset.isnull().count())
    missing_data = pd.concat([total_missing,percent_misiing],axis=1,keys = ['Total','Percent'])
    return missing_data
    
def drop_missing(dataset,threshold):
    '''
    drop feature from dataset if percentage of missing value is greater than threshold

    Parameters
    ----------
    dataset : Dataframe
        DESCRIPTION.
    threshold : int32
        DESCRIPTION.

    Returns
    -------
    dataset : Dataframe
        DESCRIPTION.

    '''
    missing = missing_data(df)
    dataset.drop(columns = missing[missing['Percent']>threshold].index)
    print(dataset.isnull().sum().sort_values(ascending = False))
    return dataset

def impute_missing(dataset):
    '''
    Impute numerical features with mean of the feature and categorical features
    with mode of the feature and return dataframe

    Parameters
    ----------
    dataset : Dataframe
        DESCRIPTION.

    Returns
    -------
    dataset : Dataframe
        DESCRIPTION.

    '''
    num = num_features(dataset)
    cat = cat_features(dataset)
    mean_imputer = SimpleImputer(strategy = 'mean')
    num = pd.DataFrame(data= mean_imputer.fit_transform(num),columns = num.columns.tolist())
    mode_imputer = SimpleImputer(strategy = 'most_frequent')
    cat = pd.DataFrame(data= mode_imputer.fit_transform(cat),columns = cat.columns.tolist())
    dataset = pd.concat([num,cat],axis = 1)
    return dataset

def check_imbalance(target):
    '''
    calculate class imbalance for target variable

    Parameters
    ----------
    target : Series
        DESCRIPTION.

    Returns
    -------
    class_values : Series
        DESCRIPTION.

    '''
    class_values = (target.value_counts()/target.value_counts().sum())*100
    return class_values

def generate_pair_plot(dataset):
    '''
    Generate pair plot for the features in dataset

    Parameters
    ----------
    dataset : Dataframe
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    plt.figure(figsize = (20,20))
    sns.pairplot(data = dataset,hue = 'Claim',palette = sns.color_palette(palette = 'Set2'),diag_kind = 'kde')
    plt.legend(df['Claim'].unique())
    plt.show()
    
def one_hot_encode(dataset):
    '''
    OneHotEncode Categorical features and drop first class for each feature

    Parameters
    ----------
    dataset : Dataframe
        DESCRIPTION.

    Returns
    -------
    dataset : Dataframe
        DESCRIPTION.

    '''
    dataset = pd.get_dummies(dataset,drop_first = True)
    return dataset

def std_scaler(dataset):
    '''
    Scale numerical features with mean =0 and Std. deviation =1

    Parameters
    ----------
    dataset : Dataframe
        DESCRIPTION.

    Returns
    -------
    dataset : Dataframe
        DESCRIPTION.

    '''
    scaler = StandardScaler()
    dataset = pd.DataFrame(scaler.fit_transform(dataset),columns=dataset.columns.tolist(),index = dataset.index.tolist())
    return dataset

def lbl_encoder(dataset):
    '''
    label encode categorical features

    Parameters
    ----------
    dataset : Dataframe
        DESCRIPTION.

    Returns
    -------
    dataset : Dataframe
        DESCRIPTION.

    '''
    label = LabelEncoder()
    #dataset = pd.DataFrame(label.fit_transform(dataset),columns=dataset.columns.tolist(),index = dataset.index.tolist())
    cat_columns = dataset.select_dtypes(include='object').columns.tolist()
    try:
        for i in cat_columns:
            dataset[i] = label.fit_transform(dataset[i])
    except :
        print('Encoding error ',i)
    #print(dataset)
    return dataset

def build_dataset(dataset,encode_type = 0):
    '''
    OneHotEncode categorical data if encode type = 1
    Label Encode categorical data if encode type = 0

    Parameters
    ----------
    dataset : Dataframe
        DESCRIPTION.
    encode_type : int32, optional
        DESCRIPTION. The default is 0.

    Returns
    -------
    ds_new : Dataframe
        DESCRIPTION.

    '''
    try:
        num_feature = num_features(dataset)
        cat_feature = cat_features(dataset) 
        if encode_type == 1:
            num = std_scaler(num_feature)
            cat = one_hot_encode(cat_feature)
        else:
            num = num_feature
            cat = lbl_encoder(cat_feature)

        ds_new = pd.concat([num,cat],axis = 1)
        return ds_new
    except Exception as e:
         print('Build Dataset failed \n',+str(e.message()))
         
def run_model(dataset,target,model):
    '''
    Build dataset based on distance based model or tree based model.
    Split data into test and train with test size of 20%.
    Fit the model to training set.
    Predict the output on testing set.
    Generate Classification report, Confusion Matrix, Accuracy score.
    Plot Area under Curve for the model.
    Parameters
    ----------
    dataset : Dataframe
        DESCRIPTION.
    target : Series
        DESCRIPTION.
    model : TYPE
        models from sklearn package.

    Returns
    -------
    None.

    '''
    'for Distance Based models'
    if (model.__class__ == LogisticRegression) or (model.__class__ == KNeighborsClassifier):      
        X_new = build_dataset(dataset,1)
    else:
        X_new = build_dataset(dataset)         
    print(X_new.shape)
    X_train,X_test,y_train,y_test = train_test_split(X_new,target,random_state = 42, test_size =0.2,stratify = target)
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    print('===='*20)
    print(type(model))
    print('===='*20)
    print('Classification Report : \n',metrics.classification_report(y_test,y_pred))
    print('Confusion Matrix : \n',metrics.confusion_matrix(y_test,y_pred))
    print('Accuracy score: \n',metrics.accuracy_score(y_test, y_pred))
    fpr,tpr,threshold = metrics.roc_curve(y_test,y_pred)
    plt.plot(fpr, tpr)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC curve')
    plt.show()
    print('AUC : ',metrics.roc_auc_score(y_test, y_pred))
    return model,X_train.columns.tolist()
      
def feature_importance_rfe(dataset,target,model):
    '''
    Generate feature importance using Recursive feature elimination 

    Parameters
    ----------
    dataset : Dataframe
        DESCRIPTION.
    target : Series
        DESCRIPTION.
    model : TYPE
        models from sklearn package.
    Returns
    -------
    None.

    '''
    'for Distance Based models'
    try:
        if model.__class__ == LogisticRegression :      
            X_new = build_dataset(dataset,1)
        else:
            X_new = build_dataset(dataset) 
        rfe = RFECV(estimator = model,min_features_to_select = 5,cv=10)
        rfe.fit(X_new,target)
        fet_imp = pd.Series(rfe.ranking_,index = X_new.columns.tolist())
        print('===='*20)
        print('Feature Importance with model',type(model))
        print('===='*20)
        print(fet_imp.sort_values(ascending=True))
    except Exception as e:
        print('feature_importance_rfe failed : \n'+str(e.message()))
    
def feature_importance_tree_based_models(dataset,target,model):
    '''
    Generate feature importance using Tree Based Models

    Parameters
    ----------
    dataset : Dataframe
        DESCRIPTION.
    target : Series
        DESCRIPTION.
    model : TYPE
        Random Forest Classifier from sklearn.ensemble package.

    Returns
    -------
    None.

    '''
    try:
        X_new = build_dataset(dataset) 
        print(X_new.head())
        model.fit(X_new,target)
        feature_importance = pd.Series(model.feature_importances_,index = X_new.columns.tolist())
        print('===='*20)
        print('Feature Importance with model',type(model))
        print('===='*20)
        feature_importance = feature_importance.sort_values(ascending = False)
        print(feature_importance.sort_values(ascending = False))
    except Exception as e:
        print('feature_importance_tree_based_models failed : \n'+str(e.message()))
    
def grid_search_random_forest(dataset,target,params,kfolds = 10): 
    '''
    Find Best Random forest model for the given dataset.
    Need to run only once to get the best paramneters.

    Parameters
    ----------
    dataset : Dataframe
        DESCRIPTION.
    target : Series
        DESCRIPTION.
    params : Dict
        DESCRIPTION.
    kfolds : Int32, optional
        DESCRIPTION. The default is 10.

    Returns
    -------
    model : RandomForestClassifier()
        DESCRIPTION.

    '''
    try:
        X_new = build_dataset(dataset) 
        rfc = RandomForestClassifier(n_estimators = 100,class_weight = 'balanced',random_state = 42)
        model = GridSearchCV(estimator = rfc,param_grid = params,cv = kfolds)
        model.fit(X_new,target)
        print('Best Estimator :\n',model.best_estimator_)
        print('Best Score : ',model.best_score_)
        return model
    except Exception as e:
        print('grid_search_random_forest failed : \n',+str(e))
        
def grid_search(dataset,target,params,model,kfolds = 10):
    try:
        if (model.__class__ == LogisticRegression) or (model.__class__ == KNeighborsClassifier):      
            X_new = build_dataset(dataset,1)
        else:
            X_new = build_dataset(dataset)    
        gscv = GridSearchCV(estimator = model,param_grid = params,cv = kfolds)
        gscv.fit(X_new,target)
        return gscv
    except Exception as e:
        print('grid_search failed : \n',str(e.message()))
        
def run_model_with_smote(dataset,target,model):
    try:        
        if (model.__class__ == LogisticRegression) or (model.__class__ == KNeighborsClassifier):      
            X_new = build_dataset(dataset,1)
        else:
            X_new = build_dataset(dataset) 
        sm = SMOTE(random_state =42)        
        X_smote,y_smote = sm.fit_resample(X_new,target)
        X_train,X_test,y_train,y_test = train_test_split(X_smote,y_smote,random_state = 42, test_size =0.2,stratify = y_smote)
        print('Class ratio after applyin SMOTE : \n',check_imbalance(y_smote))
        model.fit(X_train,y_train)
        y_pred = model.predict(X_test)
        print('===='*20)
        print(type(model))
        print('===='*20)
        print('Classification Report : \n',metrics.classification_report(y_test,y_pred))
        print('Confusion Matrix : \n',metrics.confusion_matrix(y_test,y_pred))
        print('Accuracy score: \n',metrics.accuracy_score(y_test, y_pred))
        fpr,tpr,threshold = metrics.roc_curve(y_test,y_pred)
        plt.plot(fpr, tpr)
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('ROC curve')
        plt.show()
        print('AUC : ',metrics.roc_auc_score(y_test, y_pred))
        return model,X_train.columns.tolist()
    except Exception as e :
        print('run_model_with_smote failed : \n',str(e.message()))

def boosting_with_smote(dataset,target):
    '''
    Use SMOTE to oversample minority class and find accuracy using XGBoost

    Parameters
    ----------
    dataset : Dataframe
        DESCRIPTION.
    target : Series
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    try:
        xgb = XGBClassifier(random_state = 42)
        X_new = build_dataset(dataset)
        sm = SMOTE(sampling_strategy = 'minority' ,random_state = 10)
        X_smote,y_smote = sm.fit_resample(X_new, target)
        print('Shape after SMOTE : ',X_smote.shape)
        X_train,X_test,y_train,y_test = train_test_split(X_smote,y_smote,random_state = 42, test_size =0.2,stratify = y_smote)
        xgb.fit(X_train,y_train)
        y_pred = xgb.predict(X_test)
        print('===='*20)
        print(type(xgb))
        print('===='*20)
        print('Classification Report : \n',metrics.classification_report(y_test,y_pred))
        print('Confusion Matrix : \n',metrics.confusion_matrix(y_test,y_pred))
        print('Accuracy score: \n',metrics.accuracy_score(y_test, y_pred))
        fpr,tpr,threshold = metrics.roc_curve(y_test,y_pred)
        plt.plot(fpr, tpr)
        plt.xlabel('FPR')
        plt.ylabel('TPR')
        plt.title('ROC curve')
        plt.show()
        print('AUC : ',metrics.roc_auc_score(y_test, y_pred))
    except Exception as e:
        print('boosting_with_smote failed : \n',+str(e.message()))
    
def submission_data(test_data,model,features,ID):
    
    '''
    fit the final model on test_data to generate predicted target.
    Merge ID and target into single Series.
    Save it as CSV file in the same directory

    Parameters
    ----------
    test_data : Dataframe
        DESCRIPTION.
    model : TYPE
        models from sklearn package..
    ID : Int32
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    try:
        if (model.__class__ == LogisticRegression) or (model.__class__ == KNeighborsClassifier):      
                X_new = build_dataset(test_data,1)
        else:
                X_new = build_dataset(test_data) 
        
        X_new = X_new.loc[:,features]
        print(X_new.head())
        y_pred = model.predict(X_new)
        #print(y_pred.head())
        submission_data = pd.Series(y_pred,index = ID,name = 'Claim')
        submission_data.to_csv('Submission.csv',header = submission_data.name,index = True)
    except Exception as e:
        print('submission_data failed : \n',str(e.message()))


def normalize_features(dataset):
    num = num_features(dataset)
    pt = PowerTransformer()
    num = pd.DataFrame(data = pt.fit_transform(num),columns = list(num))
    print('plot after transformation with lambda :',pt.lambdas_)
    for i in list(num):   
        plt.figure(figsize = (5,5))
        sns.distplot(num[i])
        plt.show()
        print('Skewness :',num[i].skew())
        print('kurtosis :',num[i].kurtosis())
    return num    
        
        
df = pd.read_csv('train.csv')
print('Shape of Dataset :', df.shape)

df.drop(columns = 'ID',inplace = True)
    
print('Feature Data Types :\n',check_dtypes(df))
print('Missing Data :\n',missing_data(df))

df = drop_missing(df, 0.6)
X = df.drop(columns = 'Claim',errors = 'ignore')
y = df['Claim']
print('Class Ratio :\n',check_imbalance(y))

#normalize_features(X)

#generate_pair_plot(df)

# numerical_data_correlation(df)

#categorical_data_chisquare(df)

# cs_Agency = pd.crosstab(index = df['Agency'], columns = df['Claim'])
# cs_Agency

# sns.barplot(x = cs_Agency.index,y = cs_Agency[1])
# sns.barplot(x = 'Agency Type',y ='Net Sales',hue = 'Claim',data=df, color='grey')
# sns.barplot(x = 'Distribution Channel',y ='Net Sales',hue = 'Claim',data=df, color='gold')
# sns.barplot(x = 'Product Name',y ='Net Sales',hue = 'Claim',data=df, color='teal')
# plt.xticks(rotation = 90)


start_time =time.time()
X_1=X.drop(index = X[X['Duration']<0].index.tolist())
#X_1 = X_1.drop(columns = ['Distribution Channel'])
X_1[X_1['Duration']<0]
y_1 = y.drop(index = X[X['Duration']<0].index.tolist())

'''To run any model use run_model function; If the model is distance based, categorical
   columns will be OneHot Encoded else if tree based model Label Encoded by the function'''
'''This model below is the final model used for optimum precision score'''


#lr = LogisticRegression()
#model,features = run_model(X_1, y_1, lr)

# dtc = DecisionTreeClassifier()
# model,features = run_model(X_1, y_1, dtc)

# rfc = RandomForestClassifier()
# model,features = run_model(X_1, y_1, rfc)


# rfc = RandomForestClassifier(class_weight = 'balanced',random_state=42)
# param={'bootstrap': [True],'max_depth':np.arange(30,36),'max_features': [2, 3,4,5,6,7],
#     'min_samples_leaf': [3, 4, 5],
#     'min_samples_split': [2,4,8, 10, 12],
#     'n_estimators': [100, 200, 300, 1000]}
# rscv = RandomizedSearchCV(estimator = rfc,param_distributions = param,cv=3)
# X_1 = build_dataset(X_1)
# rscv.fit(X_1,y)

rfc_1 = RandomForestClassifier(bootstrap=True, ccp_alpha=0.0, class_weight='balanced',
                       criterion='entropy',
                       max_leaf_nodes=None, max_samples=None,
                       min_impurity_decrease=0.0, min_impurity_split=None,
                       min_samples_split=2,
                       min_weight_fraction_leaf=0.0, n_estimators=300,
                       n_jobs=None, oob_score=False, random_state=42, verbose=0,
                       warm_start=False)
#model,features = run_model(X_1,y,rfc_1)
#model,features = run_model_with_smote(X_1,y,rfc_1)
#learning_rate = np.arange(0.1,1.1,0.1)
# for i in learning_rate:
#     gbc = GradientBoostingClassifier(n_estimators = 1000,learning_rate = i,random_state = 42)
#     model,features = run_model(X_1, y, gbc)
## learning rate = 0.6 gave max precision and accuracy
    

gbc  =GradientBoostingClassifier(n_estimators = 1000,learning_rate = 0.6,random_state = 42)
model_1 = ('rfc',rfc_1)
model_2 = ('gbc',gbc)
vc = VotingClassifier(estimators=[model_1,model_2])
model,features = run_model(X_1, y_1, vc)
test_data = pd.read_csv('test.csv')
ID = test_data['ID']
submission_data(test_data,model,features,ID)
print('Total time :',time.time() - start_time)















