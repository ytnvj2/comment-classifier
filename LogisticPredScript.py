# Import all the required libraries
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,roc_auc_score
# The function takes the content as input and outputs all the comments stripping everything (numbers,punctuations) and coverting to lower case.
def stripped_text(text):
    return " ".join(re.findall("[A-Za-z]+",text.lower()))

#Load the data and inspect
def load_data():
    df_train = pd.read_csv('Data Sets/train.csv')
    df_test = pd.read_csv('Data Sets/test.csv')
    print('Train',df_train.head())
    print('Test',df_test.head())
    print(df_train.info())
    df_train.reset_index(inplace=True)
    df_test.reset_index(inplace=True)
    # Feature Selection: Careful examination of the data reveals that the classification is purely based on the COMMENT
    # and the CLASS. So we remove the other features.
    df_train=df_train[['CONTENT','CLASS']]
    df_test=df_test[['index','CONTENT']]
    # Veiwing for the value counts to check for class imbalance.
    df_train['CLASS'].value_counts()
    # No class imbalance present, so we can move forward with pre-processing the text data
    return df_train,df_test

# Text Preprocessing
def preprocess_text(df):
    df.loc[:,'CONTENT']=df['CONTENT'].apply(stripped_text)
    # Count words and their occurences in the data
    c_vec = CountVectorizer(stop_words='english')
    X_c = c_vec.fit_transform(df['CONTENT'])
    t = TfidfTransformer()
    X_tfidf = t.fit_transform(X_c)
    return df,c_vec,t,X_tfidf

# The data has been processed, so we begin with model selection
def train(X,y,type='LR'):
    if type=='LR':
        # Using the simplest model first, LogisticRegression
        model = LogisticRegression(random_state=123)
    elif type=='RF':
        model = RandomForestClassifier(random_state=123)
    elif type=='SVM':
        model = SGDClassifier(random_state=123)
    elif type == 'KNN':
        model=KNeighborsClassifier()
    # Dividing the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=112)
    model.fit(X_train,y_train)
    return model,X_train, X_test, y_train, y_test

# Finding optimal params using cross validation
def cross_validate(X,y,type='LR'):
    if type=='LR':
        # Using the simplest model first, LogisticRegression
        parameters = {'penalty' : ['l1','l2'],'max_iter': [100,300,500],'random_state': [123]}
        model = GridSearchCV(LogisticRegression(),parameters)
    elif type=='RF':
        parameters = {
                     'max_depth' : [1,3,5],
                     'n_estimators': [50,100],
                     'max_features': ['sqrt', 'auto', 'log2'],
                     'min_samples_leaf': [1, 5, 10],
                     'bootstrap': [True, False],
                     'random_state':[123]
                     }
        model = GridSearchCV(RandomForestClassifier(),parameters)
    elif type=='SVM':
        parameters ={
             'loss' : ['hinge'],
             'penalty': ['l1','l2','elasticnet'],
             'learning_rate': ['optimal', 'invscaling'],
             'eta0':[0.01,0.1,0.001,0.0001],
             'random_state': [123]
            }
        model = GridSearchCV(SGDClassifier(),parameters)
    elif type=='KNN':
        parameters ={
             'n_neighbors' : [3,5,7,9,11],
             'p': [1,2],
            }
        model = GridSearchCV(KNeighborsClassifier(),parameters)
    #
    # Dividing the data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=112)
    model.fit(X_train,y_train)
    return model,X_train, X_test, y_train, y_test

# Model Evaluation using sklearn metrics
def evaluate_model(model,X_train, X_test, y_train, y_test):
    predictions = model.predict(X_train)
    print('Train Evaluation')
    print(confusion_matrix(y_train,predictions))
    print(roc_auc_score(y_train,predictions))
    print(classification_report(y_train,predictions))
    predictions = model.predict(X_test)
    print('Test Evaluation')
    print(confusion_matrix(y_test,predictions))
    print(roc_auc_score(y_test,predictions))
    print(classification_report(y_test,predictions))

# Generating submission from test data
def predict_class(df,c_vec,t,model,filename):
    comment=df['CONTENT'].apply(stripped_text)
    comment_c=c_vec.transform(comment)
    comment_tfidf=t.transform(comment_c)
    preds = model.predict(comment_tfidf)
    df['CLASS']=preds
    df.columns=['ID','CONTENT','CLASS']
    df.to_csv(filename,columns=['ID','CLASS'],header=True,index=False)

if __name__=='__main__':
    tr,te=load_data()
    tr,c_vec,t,X=preprocess_text(tr)
    print('Logistic Regression')
    model,X_train, X_test, y_train, y_test=train(X,tr['CLASS'])
    evaluate_model(model,X_train, X_test, y_train, y_test)
    predict_class(te,c_vec,t,model,'./preds_lr.csv')
    print('Cross-Validation Results')
    model,X_train, X_test, y_train, y_test=cross_validate(X,tr['CLASS'])
    evaluate_model(model,X_train, X_test, y_train, y_test)
    predict_class(te,c_vec,t,model,'./preds_lr_cv.csv')
    print('Random Forest Classification')
    model,X_train, X_test, y_train, y_test=train(X,tr['CLASS'],'RF')
    evaluate_model(model,X_train, X_test, y_train, y_test)
    predict_class(te,c_vec,t,model,'./preds_rf.csv')
    print('Cross-Validation Results')
    model,X_train, X_test, y_train, y_test=cross_validate(X,tr['CLASS'],'RF')
    evaluate_model(model,X_train, X_test, y_train, y_test)
    predict_class(te,c_vec,t,model,'./preds_rf_cv.csv')
    print('SVM Classifier')
    model,X_train, X_test, y_train, y_test=train(X,tr['CLASS'],'SVM')
    evaluate_model(model,X_train, X_test, y_train, y_test)
    predict_class(te,c_vec,t,model,'./preds_svm.csv')
    print('Cross-Validation Results')
    model,X_train, X_test, y_train, y_test=cross_validate(X,tr['CLASS'],'SVM')
    evaluate_model(model,X_train, X_test, y_train, y_test)
    predict_class(te,c_vec,t,model,'./preds_svm_cv.csv')
    print('KNN Classifier')
    model,X_train, X_test, y_train, y_test=train(X,tr['CLASS'],'KNN')
    evaluate_model(model,X_train, X_test, y_train, y_test)
    predict_class(te,c_vec,t,model,'./preds_knn.csv')
    print('Cross-Validation Results')
    model,X_train, X_test, y_train, y_test=cross_validate(X,tr['CLASS'],'KNN')
    evaluate_model(model,X_train, X_test, y_train, y_test)
    predict_class(te,c_vec,t,model,'./preds_knn_cv.csv')
