#!/usr/bin/env python
# coding: utf-8

# # Importing Libraries

# In[ ]:


import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GRU
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import SpatialDropout1D
from keras.models import Sequential
from keras.models import load_model
from keras import layers
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import GRU
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.initializers import Constant
from keras.callbacks import ModelCheckpoint

import gensim
from gensim.models import KeyedVectors
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import accuracy_score


# # Importing Dataset

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


data = pd.read_csv('drive/My Drive/ThirdYearProject/KrakenDataset.csv')
data['Sentiment'].value_counts()
data.head


# # Tweet Preprocessing

# In[ ]:


# encode the three sentiments to three distinct numbers for model using lambda
#def convert(sentiment): {'Positive' : 0, 'Negative' : 1, 'Neutral' : 2}

def convert(sentiment):
    if sentiment == 'Positive':
        return 0
    elif sentiment == 'Negative':
        return 1
    elif sentiment == 'Neutral':
        return 2

# for the coloumn Sentiment, apply the function 'convert' to it
data['Sentiment'] = data['Sentiment'].apply(convert)
data.head


# In[ ]:


# data split into 70% training and 30% testing
validationSplit = 0.3

# specify the maximum length of the tweet, which is 280 characters, if less it will be
# padded 280 characters translates to 55 words as 5.1 characters are average word length
tweetLength = 55

# create a tokenizor without a maximum number of unique words
# oov is out of vocabulary token for any new words encountered during testing
tokenizer = Tokenizer(oov_token = 'Unknown')

# create a vocabulary based on word frequency: wordIndex['word'] = unique value.
tokenizer.fit_on_texts(data['Tweets'])

# each word is indexed to map the vocabulary to their corresponding numbers
wordIndex = tokenizer.word_index
# encode tweets into sequences of mapped vocabulary of tokens, where each number in the
# sequence corresponds to a word
trainingSequences = tokenizer.texts_to_sequences(data['Tweets'])

vocabularySize = len(wordIndex) + 1
print(vocabularySize)
longestTweet = max([len(tweet) for tweet in trainingSequences])
print(longestTweet)

# padding any tweet less than 280 characters with 0s at the end to acheive uniform length
tweetsPadded = pad_sequences(trainingSequences, maxlen = tweetLength, padding = 'post')
sentiments = data['Sentiment'].values

# Output results
print("Word index:\n", wordIndex)
print("\nTraining tweet sequences:\n", trainingSequences)
print("\nPadded tweet training sequences:\n", tweetsPadded)
print("\nPadded training sequences shape:", tweetsPadded.shape)
print("data type:", type(trainingSequences))
print("Padded data type:", type(tweetsPadded))


# # Model Setup

# In[ ]:


# shuffle dataset in order for model to achieve generalization
# shuffle by introducing indices so original form is retained


# In[ ]:


numberOfTestTweets = int(validationSplit * tweetsPadded.shape[0])
Y = sentiments

trainingTweets = tweetsPadded[:- numberOfTestTweets]
trainingY = Y[:- numberOfTestTweets]

testingTweets = tweetsPadded[-numberOfTestTweets:]
testingY = Y[-numberOfTestTweets:]


# In[ ]:


print('trainingTweets shape:', trainingTweets.shape)
print('trainingY shape:', trainingY.shape)

print('testingTweets shape:', testingTweets.shape)
print('testingY shape:', testingY.shape)


# In[ ]:



def printResults(model, trainedModel):
  accuracy = trainedModel.history['accuracy']
  val_accuracy = trainedModel.history['val_accuracy']
  loss = trainedModel.history['loss']
  val_loss = trainedModel.history['val_loss']

  epochs = range(1, len(accuracy)+1)

  plt.plot(epochs, accuracy, 'g', label='Training accuracy')
  plt.plot(epochs, val_accuracy, 'r', label='Validation accuracy')
  plt.title('Training and validation accuracy')
  plt.legend()

  plt.figure()

  plt.plot(epochs, loss, 'g', label='Training loss')
  plt.plot(epochs, val_loss, 'r', label='Validation loss')
  plt.title('Training and validation loss')
  plt.legend()

  plt.show()
  # plt functions taken from: https://towardsdatascience.com/deep-transfer-learning-for-image-classification-f3c7e0ec1a14
  prediction = model.predict(testingTweets)
  prediction = np.argmax(prediction, axis=1)

  classificationReport = classification_report(testingY, prediction.round())
  print(classificationReport)

  confusionMatrix = confusion_matrix(testingY, prediction.round())
  print('Confusion Matrix:')
  print(confusionMatrix)

  print('Áccuracy Score:')
  print(accuracy_score(testingY, prediction.round()))


# # Word2Vec Word Embeddings

# In[ ]:


# using a pre-trained embedding on the tweets from google
word2Vecmodel = KeyedVectors.load_word2vec_format('drive/My Drive/ThirdYearProject/GoogleNews-vectors-negative300.bin', binary=True)


# In[ ]:


# create an embedding matrix where all the words not in word2Vec model dictionary are zeroed
# the embedding vector dimension is 300 as in the file
embeddingMatrix = np.zeros((len(wordIndex)+1, 300))
for word, w in wordIndex.items():
    if word in word2Vecmodel:
        embeddingVector = word2Vecmodel[word]
        embeddingMatrix[w, :] = embeddingVector

# using keras built-in embedding layer
# maps words to their corresponding embedding vectors in embeddingMatrix
embeddingLayer = Embedding(len(wordIndex) + 1, 300,
                           weights = [embeddingMatrix],
                           input_length = tweetLength,
                           trainable = False)

# code function from: https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/


# # CNN Model - Word2Vec

# In[ ]:


CNNmodel = Sequential()
CNNmodel.add(embeddingLayer)
CNNmodel.add(layers.GlobalMaxPool1D())
CNNmodel.add(layers.Dense(10, activation = 'relu', name='layer2'))
CNNmodel.add(layers.Dense(3, activation = 'softmax', name='layer3'))
CNNmodel.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
               metrics = ['accuracy'])
CNNmodel.summary()


# In[ ]:


trainedCNNmodel = CNNmodel.fit(trainingTweets, trainingY, batch_size = 32, epochs = 25, validation_data=(testingTweets, testingY), verbose = 1 )


# In[ ]:


printResults(CNNmodel, trainedCNNmodel)


# # LSTM Model - Word2Vec

# In[ ]:


LSTMmodel = Sequential()
LSTMmodel.add(embeddingLayer)
LSTMmodel.add(LSTM(units=32, dropout=0.2, recurrent_dropout=0.25))
LSTMmodel.add(Dense(3, activation='softmax'))

LSTMmodel.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics = ['accuracy'])

LSTMmodel.summary()


# In[ ]:


trainedLSTMmodel = LSTMmodel.fit(trainingTweets, trainingY, batch_size = 32, epochs = 25, validation_data=(testingTweets, testingY), verbose = 1 )


# In[ ]:


printResults(LSTMmodel, trainedLSTMmodel)


# # BiLSTM Model - Word2Vec

# In[ ]:


BiLSTMmodel = Sequential()
BiLSTMmodel.add(embeddingLayer)
BiLSTMmodel.add(SpatialDropout1D(0.2))
BiLSTMmodel.add(Bidirectional(GRU(128)))
BiLSTMmodel.add(Dense(128, activation='relu'))
BiLSTMmodel.add(Dropout(0.2))
BiLSTMmodel.add(Dense(3, activation='softmax'))

BiLSTMmodel.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
BiLSTMmodel.summary()


# In[ ]:


trainedBiLSTMmodel = BiLSTMmodel.fit(trainingTweets, trainingY, batch_size = 32, epochs = 25, validation_data=(testingTweets, testingY), verbose = 1 )


# In[ ]:


printResults(BiLSTMmodel, trainedBiLSTMmodel)


# # GloVe Word Embeddings

# In[ ]:


gloVeEmbeddingIndex = {}
pretrainedEmbeddings = open('drive/My Drive/ThirdYearProject/glove.twitter.27B.100d.txt', encoding = 'utf-8')
for line in pretrainedEmbeddings:
  values = line.split()
  word = values[0]
  coeff = np.asarray(values[1:], dtype='float32')
  gloVeEmbeddingIndex[word] = coeff
pretrainedEmbeddings.close()

# function taken from: https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/
# In[ ]:


# create an embedding matrix where all the words not in word2Vec model dictionary are zeroed
# the embedding vector dimension is 100 as in the file
embeddingMatrix = np.zeros((len(wordIndex)+1, 100))
for word, w in wordIndex.items():
  embeddingVector = gloVeEmbeddingIndex.get(word)
  if embeddingVector is not None:
    embeddingMatrix[w, :] = embeddingVector

# using keras built-in embedding layer
# maps words to their corresponding embedding vectors in embeddingMatrix
embeddingLayer = Embedding(len(wordIndex) + 1, 100,
                           weights = [embeddingMatrix],
                           input_length = tweetLength,
                           trainable = False)

# code function from: https://machinelearningmastery.com/use-word-embedding-layers-deep-learning-keras/


#
#
# # CNN Model - GloVe
#
#

# In[ ]:


CNNmodel = Sequential()
CNNmodel.add(embeddingLayer)
CNNmodel.add(layers.GlobalMaxPool1D())
CNNmodel.add(layers.Dense(10, activation = 'relu', name='layer2'))
CNNmodel.add(layers.Dense(3, activation = 'softmax', name='layer3'))
CNNmodel.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
               metrics = ['accuracy'])
CNNmodel.summary()


# In[ ]:


trainedCNNmodel = CNNmodel.fit(trainingTweets, trainingY, batch_size = 32, epochs = 25, validation_data=(testingTweets, testingY), verbose = 1 )


# In[ ]:


printResults(CNNmodel, trainedCNNmodel)


# # LSTM Model - GloVe

# In[ ]:


LSTMmodel = Sequential()
LSTMmodel.add(embeddingLayer)
LSTMmodel.add(LSTM(units=32, dropout=0.2, recurrent_dropout=0.25))
LSTMmodel.add(Dense(3, activation='softmax'))

LSTMmodel.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam',
                  metrics = ['accuracy'])

LSTMmodel.summary()


# In[ ]:


trainedLSTMmodel = LSTMmodel.fit(trainingTweets, trainingY, batch_size = 32, epochs = 25, validation_data=(testingTweets, testingY), verbose = 1 )


# In[ ]:


printResults(LSTMmodel, trainedLSTMmodel)


# # BiLSTM Model - GloVe

# In[ ]:


BiLSTMmodel = Sequential()
BiLSTMmodel.add(embeddingLayer)
BiLSTMmodel.add(SpatialDropout1D(0.2))
BiLSTMmodel.add(Bidirectional(GRU(128)))
BiLSTMmodel.add(Dense(128, activation='relu'))
BiLSTMmodel.add(Dropout(0.2))
BiLSTMmodel.add(Dense(3, activation='softmax'))

BiLSTMmodel.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
BiLSTMmodel.summary()


# In[ ]:


trainedBiLSTMmodel = BiLSTMmodel.fit(trainingTweets, trainingY, batch_size = 32, epochs = 25, validation_data=(testingTweets, testingY), verbose = 1 )


# In[ ]:


printResults(BiLSTMmodel, trainedBiLSTMmodel)
