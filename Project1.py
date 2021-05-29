#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import RegexpTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


# In[18]:


data = pd.read_csv("1429_1.csv")


# In[19]:


data.head()


# In[20]:


data = data[['reviews.rating', 'reviews.text']]
data = data.dropna()
data.head()


# In[5]:


data['senti'] = data['reviews.rating']>=4
data['senti'] = data['senti'].replace([True, False] , ['pos', 'neg'])


# In[7]:


data['senti'].value_counts().plot.bar()


# In[21]:


data2 = pd.read_csv("Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv")
data2 = data2[['reviews.rating' , 'reviews.text']]
data2 = data2[data2['reviews.rating']<=3]

data3 = pd.read_csv("Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products.csv")
data3 = data3[['reviews.rating' , 'reviews.text']]
data3 = data3[data3['reviews.rating']<=3]

frames = [data, data2, data3]
df = pd.concat(frames)
df = df.dropna()


# In[22]:


df


# In[14]:


df.drop(['senti'], axis =1)


# In[23]:


df['senti'] = df['reviews.rating']>=4
df['senti'] = df['senti'].replace([True, False] , ['pos', 'neg'])


# In[24]:


df


# In[25]:


df['senti'].value_counts().plot.bar()


# In[26]:


import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
import re
import string
import nltk
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


# In[27]:


cleanup_re = re.compile('[^a-z]+')
def cleanup(sentence):
    sentence = str(sentence)
    sentence = sentence.lower()
    sentence = cleanup_re.sub(' ',sentence).strip()
    return sentence


# In[28]:


df['Summary_Clean'] = df['reviews.text'].apply(cleanup)


# In[29]:


df


# In[30]:


split = df[['Summary_Clean','senti']]
trainData = split.sample(frac=0.8,random_state=200)
testData = split.drop(trainData.index)


# In[31]:


trainData


# In[32]:


testData


# In[33]:


from wordcloud import STOPWORDS
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer


# In[35]:


stopwords = set(STOPWORDS)
stopwords.remove("not")

count_vect = CountVectorizer(min_df = 2 , stop_words = stopwords, ngram_range=(1,2))
tfidf_transformer = TfidfTransformer()

X_train_counts = count_vect.fit_transform(trainData["Summary_Clean"])
train_vectors = tfidf_transformer.fit_transform(X_train_counts)

X_new_counts = count_vect.transform(testData["Summary_Clean"])
test_vectors = tfidf_transformer.transform(X_new_counts)


# In[52]:


from sklearn import svm
from sklearn.metrics import classification_report

classifier_linear = svm.SVC(kernel='linear',probability=True)

classifier_linear.fit(train_vectors, trainData['senti'])

prediction_linear = classifier_linear.predict(test_vectors)


print(classification_report(testData['senti'], prediction_linear))


# In[75]:


S = classifier_linear.predict_proba(test_vectors)[:,1]

SS = classifier_linear.score(test_vectors, testData['senti'])

print("Support Vector Machines Accuracy : %f " % (SS))


# In[42]:


from sklearn.naive_bayes import MultinomialNB

model1 = MultinomialNB().fit(train_vectors , trainData["senti"])

prediction_M = model1.predict(test_vectors)

print(classification_report(testData['senti'], prediction_M))


# In[74]:


l = model1.predict_proba(test_vectors)[:,1]

ls = model1.score(test_vectors, testData['senti']) 

print("Multinomial Accuracy : %f" %(ls))


# In[44]:


from sklearn.naive_bayes import BernoulliNB

model2 = BernoulliNB()

model2.fit(train_vectors , trainData["senti"])

prediction_B = model1.predict(test_vectors)

print(classification_report(testData['senti'], prediction_B))


# In[73]:


B = model2.predict_proba(test_vectors)[:,1]

BS= model2.score(test_vectors, testData['senti'])

print("Bernoulli Accuracy : %f" %(BS))


# In[46]:


from sklearn import linear_model

logreg = linear_model.LogisticRegression(solver='lbfgs' , C=1000)

logreg.fit(train_vectors , trainData["senti"])

prediction_L = logreg.predict(test_vectors)

print(classification_report(testData['senti'], prediction_L))


# In[71]:


M = logreg.predict_proba(test_vectors)[:,1]

LG = logreg.score(test_vectors, testData['senti'])

print("Logistic Regression Accuracy : %f " %(LG))


# In[53]:


from sklearn.ensemble import RandomForestClassifier

RFC = RandomForestClassifier()

RFC.fit(train_vectors , trainData["senti"])

prediction_RFC = logreg.predict(test_vectors)

print(classification_report(testData['senti'], prediction_RFC))


# In[70]:


R = RFC.predict_proba(test_vectors)[:,1]

SC = RFC.score(test_vectors, testData['senti'])

print("Random Forest Accuracy : %f " % (SC))


# In[47]:


def test_sample(model, sample):
    sample_counts = count_vect.transform([sample])
    sample_tfidf = tfidf_transformer.transform(sample_counts)
    result = model.predict(sample_tfidf)[0]
    prob = model.predict_proba(sample_tfidf)[0]
    #print("sample estimated as %s: negative prob %f, positive prob %f" % (result.upper(), prob[0], prob[1]))
    if result.upper()=="POS":
        return 1;
    else:
        return 0;


# In[48]:


test_sample(logreg, "The product was good and easy to use")


# In[49]:


test_sample(model1, "The product was good and easy to use")


# In[54]:


test_sample(classifier_linear, "The product was good and easy to use")


# In[55]:


test_sample(RFC, "The product was good and easy to use")


# In[87]:


test_sample(classifier_linear, "Worst")


# In[79]:


X = ['RandomForestClassifier','LogisticRegression','BernoulliNB','MultinomialNB','SupportVectorMachine']
Y = [SC,LG,BS,ls,SS]

plt.xlabel("Algorithms")
plt.ylabel("Accuracy")
plt.scatter(X,Y)

plt.show()


# In[ ]:
def test_sample_csv(model, sample):
    sample_counts = count_vect.transform([sample])
    sample_tfidf = tfidf_transformer.transform(sample_counts)
    result = model.predict(sample_tfidf)[0]
    return result

def csv_predict(f):
    dfp1 = pd.read_csv(f)
    lst=[]
    for i in dfp1['reviews.text']:
        res = test_sample_csv(logreg, i)
        lst.append(res)
    Prediction=["Prediction"]    
    dfp = pd.DataFrame(lst,columns=Prediction)
    dfp1 = pd.concat([dfp1,dfp],axis=1)
    print(dfp1.head())
    dfp1.to_csv('Predicted.csv')