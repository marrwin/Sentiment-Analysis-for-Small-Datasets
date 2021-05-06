#!/usr/bin/env python
# coding: utf-8

# # Import Tweets

# In[1]:


import pandas as pd


# In[2]:


data = pd.read_csv('BumbleDataset.csv')
data.head()


# # Importing Libraries

# In[3]:


# machine learning models
from sklearn.neighbors import KNeighborsClassifier  # KNN model
from sklearn.naive_bayes import MultinomialNB       # Multi Naive Bayes
from sklearn.svm import SVC                   # Support Vector Machine 
from sklearn.linear_model import LogisticRegression # logistic regression
from sklearn.tree import DecisionTreeClassifier     # Decision Tree Classifier

from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier # meta-estimator ensemble algorithms, random forest and Adaboost


# In[4]:


# feature extraction techniques
# convert text data into nomadic form for machine learning

# CountVectorizer computes the term frequency in the data
from sklearn.feature_extraction.text import CountVectorizer

# TFIDF
from sklearn.feature_extraction.text import TfidfTransformer # Bag of word approach


# In[5]:


# sklearn train and test data split library
from sklearn.model_selection import train_test_split


# In[6]:


# machine learning pipeline to eliminate code repitition
from sklearn.pipeline import Pipeline


# In[7]:


# machine learning performance evaluation metric libraries
from sklearn.metrics import classification_report 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score 
from sklearn.metrics import roc_auc_score
from sklearn.metrics import plot_roc_curve


# In[8]:


import numpy as np
from confusion_matrix_master.cf_matrix import make_confusion_matrix
# from confusion_matrix_master.plot_CF import plot_confusion_matrix
import matplotlib.pyplot as plt


# In[9]:


# import warnings filter
from warnings import simplefilter
# ignore future warnings
simplefilter(action = 'ignore', category=FutureWarning)


# # Model Setup

# In[10]:


# selecting the two coloumns that are used for sentiment classification
data = data.loc[:, ['Tweets', 'Sentiment']]

# Independent Variable Coloumn - feature
tweets = data['Tweets'].values
# Dependent Variable Coloumn
sentiments = data['Sentiment'].values

# split training and testing data into 30% testing and 70% training, with randomness
tweets_training, tweets_testing, sentiments_training, sentiments_testing = train_test_split(tweets, sentiments, test_size=0.3, random_state=123)


# In[11]:


# Result Printing Function

def printResults(model, modelName, realvalues, prediction):
    print(modelName)
    print('Accuracy: {}%'.format(round(accuracy_score(realvalues, prediction)*100,2)))
    print('Confusion Matrix: \n')
    #confusion_matrix(y_true, y_pred)
    confusionMatrix = confusion_matrix(realvalues, prediction, labels=['Negative', 'Neutral', 'Positive'])
    print(confusionMatrix)
    print('Classification Report: \n')
    print(classification_report(realvalues, prediction))

    plot_confusion_matrix(model, tweets_testing, sentiments_testing, cmap='Blues' )
    plt.show()
    
    #ROCAUCscore = roc_auc_score(y_true = realvalues, y_score = model.predict_proba(tweets_testing), average='macro', multi_class='ovr')
    #print('ROC-AUC Score: \n')
    #print(ROCAUCscore)
    
    #roc_auc_score(realvalues, prediction)
    #plot_roc_curve(model, tweets_testing, sentiments_testing)
    #plt.show()


    #  'viridis', 'plasma', 'inferno', 'magma', 'cividis'
    #group_names = ['True Neg','False Pos','False Neg','True Pos']
    #'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
    #'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
    #'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn'
    #ConfusionMatrixDisplay.plot_confusion_matrix(estimator=model, X=realvalues, y_true=prediction, labels= group_names, normalize=False)
    #plt.show()
    #metrics.f1_score(realvalues, prediction, average='weighted', labels = np.unique(prediction)))
    #group_names = ['True Neg','False Pos','False Neg','True Pos']
    #group_counts = ["{0:0.0f}".format(value) for value in confusionMatrix.flatten()]
    #group_percentages = ["{0:.2%}".format(value) for value in confusionMatrix.flatten()/np.sum(confusionMatrix)]
    #labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
    #labels = np.asarray(labels).reshape(2,2)
    #sns.heatmap(confusionMatrix, annot=labels, fmt='', cmap='Blues')
    
    #group_names = ['True Neg','False Pos','False Neg','True Pos']
    #make_confusion_matrix(confusionMatrix, figsize=(8,6), cbar=False)
    
    #np.set_printoptions(precision=2)
    #plt.figure()
    #plot_confusion_matrix(confusionMatrix, classes=['Negative', 'Neutral', 'Positive'], title = 'Confusion Matrix')
    


# # KNN Model

# In[35]:


KNNpipe = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('KNNmodel', KNeighborsClassifier(n_neighbors=15, weights = 'distance'))])
#training
model = KNNpipe.fit(tweets_training, sentiments_training)
#testing
KNNprediction = model.predict(tweets_testing)


# In[36]:


printResults(model, 'KNN Model', sentiments_testing, KNNprediction)


# # Logistic Regression Model

# In[43]:


LRpipe = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), 
                   ('LogisticRegressionModel', LogisticRegression(solver = 'lbfgs'))])
#training
model = LRpipe.fit(tweets_training, sentiments_training)
#testing
LRprediction = model.predict(tweets_testing)


# In[44]:


printResults(model, 'Logistic Regression Model', sentiments_testing, LRprediction)


# # Decision Tree Model

# In[45]:


DTpipe = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('DecisionTreeModel', DecisionTreeClassifier(criterion = 'entropy'))])
#training
model = DTpipe.fit(tweets_training, sentiments_training)
#testing
DTprediction = model.predict(tweets_testing)


# In[46]:


printResults(model, 'Decision Tree Model', sentiments_testing, DTprediction)


# # Support Vector Machine Model

# In[59]:


SVMpipe = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('SVMmodel', SVC(C = 2, kernel = 'rbf'))])
#training
model = SVMpipe.fit(tweets_training, sentiments_training)
#testing
SVMprediction = model.predict(tweets_testing)


# In[60]:


printResults(model, 'SVM Model', sentiments_testing, SVMprediction)


# # Random Forest Model

# In[51]:


RFpipe = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('RandomForestModel', RandomForestClassifier(n_estimators =200))])
#training
model = RFpipe.fit(tweets_training, sentiments_training)
#testing
RFprediction = model.predict(tweets_testing)


# In[52]:


printResults(model, 'Random Forest Model', sentiments_testing, RFprediction)


# # Multinomial Naive Bayes Model

# In[69]:


MultinomialNBpipe = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('MultinomialNBmodel', MultinomialNB(alpha = 2.0))])
#training
model = MultinomialNBpipe.fit(tweets_training, sentiments_training)
#testing
MultinomialNBprediction = model.predict(tweets_testing)


# In[70]:


printResults(model, 'Multinomial Naive Bayes model', sentiments_testing, MultinomialNBprediction)


# # AdaBoost Model

# In[81]:


AdaBoostpipe = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()), ('AdaBoostModel', AdaBoostClassifier(learning_rate = 0.1))])
#training
model = AdaBoostpipe.fit(tweets_training, sentiments_training)
#testing
AdaBoostprediction = model.predict(tweets_testing)


# In[82]:


printResults(model, 'AdaBoost model', sentiments_testing, AdaBoostprediction)


# In[ ]:





# In[ ]:




