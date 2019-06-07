# Import all the required libraries
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from nltk import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score,roc_auc_score,roc_curve

# The function takes the content as input and outputs all 
# the comments stripping everything (numbers,punctuations) 
# and coverting to lower case.
def stripped_text(text):
    l=WordNetLemmatizer()
    tokens=[l.lemmatize(i) for i in text.split()]
    text=" ".join(tokens)
    return " ".join(re.findall("[A-Za-z]+",text.lower()))

#Load the data and preprocess the data
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
    auc=roc_auc_score(y_test,predictions)
    # calculate roc curve
    fpr, tpr, thresholds = roc_curve(y_test,predictions)
    plt.plot([0, 1], [0, 1], linestyle='--',label='ROC curve (area = %0.2f)' % auc)
    # plot the roc curve for the model
    plt.plot(fpr, tpr, marker='.')
    plt.legend()
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    # show the plot
    plt.show()

# Generating submission from test data
def predict_class(df,c_vec,t,model,filename):
    comment=df['CONTENT'].apply(stripped_text)
    comment_c=c_vec.transform(comment)
    comment_tfidf=t.transform(comment_c)
    preds = model.predict(comment_tfidf)
    df['CLASS']=preds
    df.columns=['ID','CONTENT','CLASS']
    df.to_csv(filename,columns=['ID','CLASS'],header=True,index=False)

# Predicting class from comment text
def pred_one(text,c_vec,t,model):
    comment = stripped_text(text)
    comment_c=c_vec.transform([comment])
    comment_tfidf=t.transform(comment_c)
    pred = model.predict(comment_tfidf)
    return pred

# Generate Word Cloud from text for Ads by default
def generate_wordcloud(df,c=1):
    comment_words = ''
    stopwords = set(STOPWORDS)
    # Generate comment words for non-ads
    for val in df[df.CLASS==c].CONTENT: 
        # typecaste each val to string 
        val = str(val).lower() 
        # split the comments into tokens 
        tokens = val.split() 
        for word in tokens: 
            comment_words = comment_words + word + ' '
    wordcloud = WordCloud(width = 800, height = 800, 
                    background_color ='white', 
                    stopwords = stopwords, 
                    min_font_size = 10).generate(comment_words) 
    # plot the WordCloud image                        
    plt.figure(figsize = (8, 8), facecolor = None) 
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.tight_layout(pad = 0) 
    t='Non-Ads'
    if c==1:
        t='Ads'
    plt.title(f'Word Cloud for {t}')
    plt.show() 

# Extracts the most significant features from the model's coeffiecients
def most_significant_features(model,df,c_vec):
    df.loc[:,'CONTENT']=df['CONTENT'].apply(stripped_text)
    X_c=c_vec.transform(df['CONTENT'])
    wc=pd.DataFrame(X_c.toarray().sum(axis=0).reshape(-1,1),columns=['Count'],index=c_vec.get_feature_names())
    wc['Weights']=model.coef_[0]
    wc['Exp(Weights)']=wc.Weights.apply(np.exp)
    wc=wc.sort_values(by='Exp(Weights)')
    print(wc.head(10))
    print(f'\n                 Table1')
    print(wc.tail(10))
    print(f'\n                 Table2')


if __name__=='__main__':
    tr,te=load_data()
    generate_wordcloud(tr)
    tr,c_vec,t,X=preprocess_text(tr)
    print('Logistic Regression')
    model_lr,X_train, X_test, y_train, y_test=train(X,tr['CLASS'])
    evaluate_model(model_lr,X_train, X_test, y_train, y_test)
    predict_class(te,c_vec,t,model_lr,'./preds_lr.csv')
    print('Cross-Validation Results')
    model_lr_cv,X_train, X_test, y_train, y_test=cross_validate(X,tr['CLASS'])
    evaluate_model(model_lr_cv,X_train, X_test, y_train, y_test)
    predict_class(te,c_vec,t,model_lr_cv,'./preds_lr_cv.csv')
    print('Random Forest Classification')
    model_rf,X_train, X_test, y_train, y_test=train(X,tr['CLASS'],'RF')
    evaluate_model(model_rf,X_train, X_test, y_train, y_test)
    predict_class(te,c_vec,t,model_rf,'./preds_rf.csv')
    print('Cross-Validation Results')
    model_rf_cv,X_train, X_test, y_train, y_test=cross_validate(X,tr['CLASS'],'RF')
    evaluate_model(model_rf_cv,X_train, X_test, y_train, y_test)
    predict_class(te,c_vec,t,model_rf_cv,'./preds_rf_cv.csv')
    print('SVM Classifier')
    model_svm,X_train, X_test, y_train, y_test=train(X,tr['CLASS'],'SVM')
    evaluate_model(model_svm,X_train, X_test, y_train, y_test)
    predict_class(te,c_vec,t,model_svm,'./preds_svm.csv')
    print('Cross-Validation Results')
    model_svm_cv,X_train, X_test, y_train, y_test=cross_validate(X,tr['CLASS'],'SVM')
    evaluate_model(model_svm_cv,X_train, X_test, y_train, y_test)
    predict_class(te,c_vec,t,model_svm_cv,'./preds_svm_cv.csv')
    print('KNN Classifier')
    model_knn,X_train, X_test, y_train, y_test=train(X,tr['CLASS'],'KNN')
    evaluate_model(model_knn,X_train, X_test, y_train, y_test)
    predict_class(te,c_vec,t,model_knn,'./preds_knn.csv')
    print('Cross-Validation Results')
    model_knn_cv,X_train, X_test, y_train, y_test=cross_validate(X,tr['CLASS'],'KNN')
    evaluate_model(model_knn_cv,X_train, X_test, y_train, y_test)
    predict_class(te,c_vec,t,model_knn_cv,'./preds_knn_cv.csv')
    most_significant_features(model_lr,tr,c_vec)