# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset [§29 Lect191: "NLP in Python - pt1"]
dataset = pd.read_csv('data/s29_Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Cleaning the texts [§29 Lect192: "NLP in Python - pt2"] & [§29 Lect193: "NLP in Python - pt3"]
# & [§29 Lect194: "NLP in Python - pt4"] & [§29 Lect195: "NLP in Python - pt5"]
# & [§29 Lect196: "NLP in Python - pt6"] & [§29 Lect197: "NLP in Python - pt7"]
import re                                                    # "NLP in Python - pt2"
import nltk                                                  # "NLP in Python - pt4"
nltk.download('stopwords')                                   # "NLP in Python - pt4"
from nltk.corpus import stopwords                            # "NLP in Python - pt4"
from nltk.stem.porter import PorterStemmer                   # "NLP in Python - pt5"
corpus = []                                                  # "NLP in Python - pt7"
for i in range(0, 1000):                                     # "NLP in Python - pt7"
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])  # "NLP in Python - pt2" & 'pt7'
    review = review.lower()                                  # "NLP in Python - pt3"
    review = review.split()                                  # "NLP in Python - pt4"
    ps = PorterStemmer()                                     # "NLP in Python - pt5"
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)                                # "NLP in Python - pt6"
    corpus.append(review)                                    # "NLP in Python - pt7"

# Creating the Bag of Words model [§29 Lect198: "NLP in Python - pt8"] & [§29 Lect199: "NLP in Python - pt9"]
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set [§29 Lect200: "NLP in Python - pt10"]
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to the Training set [§29 Lect200: "NLP in Python - pt10"]
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results [§29 Lect200: "NLP in Python - pt10"]
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix [§29 Lect200: "NLP in Python - pt10"]
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)