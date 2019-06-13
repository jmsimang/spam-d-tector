from scraper import get_data_from_url
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
import numpy as np
import pandas as pd

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data'
get_data_from_url(url)
filename = url.split('/')[-1]
data = pd.read_csv(filename).values

# Keep data scrambled to get a different result every time
np.random.shuffle(data)

# Select first 48 features as predictors, last column will be predicted
X = data[:, :48]
Y = data[:, -1]

# all rows up till last 100 rows
X_train = X[:-100, ] 
y_train = Y[:-100, ]

# Last 100 rows
X_test = X[-100:, ]
y_test = Y[-100:, ]

# create a model
model = MultinomialNB()
model.fit(X_train, y_train)
print(f'Naive Bayes classification rate: {round(model.score(X_test, y_test) * 100, 2)} % accuracy.')

model = AdaBoostClassifier(random_state=0)
model.fit(X_train, y_train)
print(f'AdaBoost classification rate: {round(model.score(X_test, y_test) * 100, 2)} % accuracy.')
