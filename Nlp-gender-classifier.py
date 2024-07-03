import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data=pd.read_csv(r"gender_classifier.csv",encoding="latin1")
data=pd.concat([data.gender,data.description],axis=1)
data.dropna(axis=0,inplace=True)
data.gender=[1 if i=="female" else 0 for i in data.gender]


first_description=data.description[4]
import re
description=re.sub("[^a-zA-Z]"," ",first_description)
description=description.lower()

import nltk
nltk.download("stopword")
nltk.download('wordnet')
nltk.download('punkt')
from nltk.corpus import stopwords


#description = description.split()
# Instead of using the splitting, we can use the tokenize function. 
description = nltk.word_tokenize(description)

#Stopwords
description = [ word for word in description if not word in set(stopwords.words("english"))]

#Lemmatizer
import nltk as nlp
kok=nlp.WordNetLemmatizer()
description=[kok.lemmatize(i) for i in description]
description=" ".join(description)

description_list=[]

for i in data.description:
        description = re.sub("[^a-zA-Z]"," ",i)
        description=description.lower()
        description = nltk.word_tokenize(description)
        description = [ i for i in description if not i in set(stopwords.words("english"))]
        kok=nlp.WordNetLemmatizer()
        description=[kok.lemmatize(i) for i in description]
        description=" ".join(description)
        description_list.append(description)
   
#%% Bag of Words
from sklearn.feature_extraction.text import CountVectorizer
max_words=15000
CV=CountVectorizer(max_features=max_words,stop_words="english")
sparce_matrix=CV.fit_transform(description_list).toarray()
print("Most used words".format(max_words,CV.get_feature_names()))

#%% Prediction with Naive Bayes
x=sparce_matrix
y=data.gender.values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=45)

from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
nb.fit(x_train,y_train)
y_pred=nb.predict(x_test)
print("Accuracy : {}".format(nb.score(y_pred.reshape(-1,1),y_test)))
