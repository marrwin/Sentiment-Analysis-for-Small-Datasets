#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[1]:


import pandas as pd               # pandas library for dataset operations
import numpy as np                # numpy library for numerical operations
import tweepy                     # twitter api used for scrapping
import csv                        # csv file to store data
import re                         # regular expression for text preprocessing
import matplotlib.pyplot as plt   # data visualization
from wordcloud import WordCloud   # data visualization
from textblob import TextBlob     # dataset annotation library


# # Twitter API Setup

# In[2]:


# API Credentials
consumer_key = ""
consumer_secret = ""
access_token = ""
access_secret = ""

# Create Authentication Object
auth = tweepy.AppAuthHandler(consumer_key, consumer_secret)

# Create API Object
api = tweepy.API(auth)


# # Tweet Scrapping

# In[3]:


preprocessedTweets = []

for tweet in tweepy.Cursor(api.search, q='clubhouse', lang = 'en').items(600):
    tweets_encoded = tweet.text.encode('utf-8')
    tweets_decoded = tweets_encoded.decode('utf-8')
    preprocessedTweets.append({"Tweets":tweets_decoded})

dataframe = pd.DataFrame.from_dict(preprocessedTweets)
dataframe.head(10)


# In[4]:


# save scrapped data into csv file
filename = 'ScrappedDataset-clubhouse.csv'
dataframe.to_csv(filename, encoding = 'utf-8')


# # Tweet Preprocessing

# In[5]:


# import scrapped dataset
# pd.read_csv imports csv file to DataFrame
dataset = pd.read_csv('ScrappedDataset-clubhouse.csv')
keep_coloumn = ['Tweets']
dataset = dataset[keep_coloumn]
dataset.to_csv('ScrappedDataset-clubhouse.csv', index = False)
dataset.head()


# In[6]:


# drop any missing values ie. empty tweets, hashtag-only tweets
dataset.isna().sum()
dataset.dropna(inplace = True)


# In[7]:


def cleanTweets(tweet):
    tweet = re.sub('RT[\s]+', '', tweet)        # Discarding RT for retweets
    tweet = re.sub('https?:\/\/\S+', '', tweet) # Discarding hyperlinks
    tweet = re.sub('#', '', tweet)              # Discarding '#' hash tag
    tweet = re.sub('@[A-Za-z0–9]+', '', tweet)  # Discarding '@' mentions
    return tweet

dataset['Tweets'] = dataset['Tweets'].apply(cleanTweets)

dataset.head(10)


# In[ ]:


get_ipython().system('pip install nltk')
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

dataset['Tweets'].apply(lambda x: [item for item in x if item not in stopwords.words()])
dataset.head(10)


# In[8]:


# drop the duplicate tweets
dataset = dataset.drop_duplicates(subset=['Tweets'])


# # Tweet Data Visualization

# In[10]:


# wordcloud

words = ' '.join([twts for twts in dataset['Tweets']])
wordCloud = WordCloud(height=300, random_state=21, max_font_size=140, background_color='white').generate(words)
plt.imshow(wordCloud, interpolation="bilinear")
plt.axis('off')
plt.show()


# In[12]:


positivetweets = dataset[dataset.Sentiment == 'Positive']
positivetweets = positivetweets['Tweets']
positivetweets = round( (positivetweets.shape[0] / dataset.shape[0]) * 100 , 1)
print (positivetweets)

negativetweets = dataset[dataset.Sentiment == 'Negative']
negativetweets = negativetweets['Tweets']
negativetweets = round( (negativetweets.shape[0] / dataset.shape[0]) * 100 , 1)
print(negativetweets)

dataset['Sentiment'].value_counts()


# In[13]:


dataset.isna().sum()
dataset.dropna(inplace = True)
dataset.to_csv('clubhouse.csv', index = False)
data = pd.read_csv('clubhouse.csv')
data.head(10)


# In[ ]:
