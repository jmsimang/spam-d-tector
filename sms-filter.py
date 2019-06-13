import pandas as pd
import matplotlib.pyplot as plt

# Text preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split

# Classifier models
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import AdaBoostClassifier
# For word map
from wordcloud import WordCloud

data = pd.read_csv('spam.csv', encoding='ISO-8859-1')
data = data.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis=1)
print(data.head())

# Rename columns/features
data.columns = ['labels', 'data']

# Create binary labels
data['b_labels'] = data['labels'].map({'ham': 0, 'spam': 1})

# Variables
y = data['b_labels'].values

# Calculating features
# using CountVectorizer
count_vectorizer = CountVectorizer(decode_error='ignore')
X = count_vectorizer.fit_transform(data['data'])
# using TfidVectorizer
obj = TfidfVectorizer(decode_error='ignore')
X2 = obj.fit_transform(data['data'])

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(X2, y, test_size=0.33, random_state=42)

print('Using CountVectorizer')
# Create the model, train it, print scores
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)
print('Train score NB: ', nb_model.score(X_train, y_train))
print('Test score NB: ', nb_model.score(X_test, y_test))
ada_model = AdaBoostClassifier(random_state=0)
ada_model.fit(X_train, y_train)
print('Train Score AdaBoost: ', ada_model.score(X_train, y_train))
print('Test Score AdaBoost: ', ada_model.score(X_test, y_test))

print('Using TfidVectorizer')
# Create the model, train it, print scores
nb_model = MultinomialNB()
nb_model.fit(X_train_2, y_train_2)
print('Train score NB: ', nb_model.score(X_train_2, y_train_2))
print('Test score NB: ', nb_model.score(X_test_2, y_test_2))
ada_model = AdaBoostClassifier(random_state=0)
ada_model.fit(X_train_2, y_train_2)
print('Train Score AdaBoost: ', ada_model.score(X_train_2, y_train_2))
print('Test Score AdaBoost: ', ada_model.score(X_test_2, y_test_2))


# visualise the data - see common words in label
def visualize(label):
    words = ''
    for word in data[data['labels'] == label]['data']:
        word = word.lower()
        words += word + ' '
    wordcloud = WordCloud(width=600, height=400).generate(words)
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.show()


visualize('spam')
visualize('ham')

# What are we getting wrong?
data['nb predictions'] = nb_model.predict(X)
data['ada predictions'] = ada_model.predict(X)

# What should be spam
print('\nWHAT SHOULD BE SPAM??? (MultinomialNB)\n')
sneaky_spam = data[(data['nb predictions'] == 0) & (data['b_labels'] == 1)]['data']
for msg in sneaky_spam:
    print(msg)
print('\nWHAT SHOULDN\'T BE SPAM??? (MultinomialNB)\n')
not_actually_spam = data[(data['nb predictions'] == 1) & (data['b_labels'] == 0)]['data']
for msg in not_actually_spam:
    print(msg)

print('\nWHAT SHOULD BE SPAM??? (AdaBoostClassifier)\n')
sneaky_spam = data[(data['ada predictions'] == 0) & (data['b_labels'] == 1)]['data']
for msg in sneaky_spam:
    print(msg)
print('\nWHAT SHOULDN\'T BE SPAM??? (AdaBoostClassifier)\n')
not_actually_spam = data[(data['ada predictions'] == 1) & (data['b_labels'] == 0)]['data']
for msg in not_actually_spam:
    print(msg)
"""
# What are we getting wrong?
data['nb predictions'] = nb_model.predict(X2)
data['ada predictions'] = ada_model.predict(X2)

# What should be spam
print('\nWHAT SHOULD BE SPAM???\n')
sneaky_spam = data[(data['nb predictions'] == 0) & (data['b_labels'] == 1)]['data']
for msg in sneaky_spam:
    print(msg)

# What shouldn't be spam
print('\nWHAT SHOULDN\'T BE SPAM???\n')
not_actually_spam = data[(data['nb predictions'] == 1) & (data['b_labels'] == 0)]['data']
for msg in not_actually_spam:
    print(msg)

# What should be spam
print('\nWHAT SHOULD BE SPAM???\n')
sneaky_spam = data[(data['ada predictions'] == 0) & (data['b_labels'] == 1)]['data']
for msg in sneaky_spam:
    print(msg)

# What shouldn't be spam
print('\nWHAT SHOULDN\'T BE SPAM???\n')
not_actually_spam = data[(data['ada predictions'] == 1) & (data['b_labels'] == 0)]['data']
for msg in not_actually_spam:
    print(msg)
"""