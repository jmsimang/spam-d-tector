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
Xtrain = X[:-100, ]
Ytrain = Y[:-100, ]

# Last 100 rows
Xtest = X[-100:, ]
Ytest = Y[-100:, ]

# create a model
model = MultinomialNB()
model.fit(Xtrain, Ytrain)
print(f'Classification rate for NB: {round(model.score(Xtest, Ytest) * 100, 2)} % accuracy.')

model = AdaBoostClassifier(random_state=0)
model.fit(Xtrain, Ytrain)
print(f'Classification rate for AB: {round(model.score(Xtest, Ytest) * 100, 2)} % accuracy.')
