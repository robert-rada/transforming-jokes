from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score, precision_score, recall_score
from sklearn.svm import SVC


SUBREDDIT = 'dadjokes'

df = pd.read_csv('processed_data/' + SUBREDDIT + '.csv', sep='<endoftext>', engine='python')

df_train = pd.read_csv('processed_data/' + SUBREDDIT + '_train.csv', sep='<endoftext>', engine='python')
print(df_train.head(3))

df_test = pd.read_csv('processed_data/' + SUBREDDIT + '_test.csv', sep='<endoftext>', engine='python')
print(df_test.head(3))

df_dev = pd.read_csv('processed_data/' + SUBREDDIT + '_dev.csv', sep='<endoftext>', engine='python')

print(len(df))
print(df_train.humor.value_counts())

# Limit number of training examples because training takes too long if there are too many examples
df_train = df_train[:80000]

# Convert texts to vectors using TF-IDF
feature_extraction = TfidfVectorizer()
feature_extraction.fit(df['text'].values)
x_train = feature_extraction.transform(df_train['text'].values)
x_test = feature_extraction.transform(df_test['text'].values)
x_dev = feature_extraction.transform(df_dev['text'].values)

y_train = df_train['humor'].values
y_test = df_test['humor'].values
y_dev = df_dev['humor'].values

print(len(y_train))

# Training SVM
print('Fitting...')
clf = SVC(probability=True, kernel='rbf', verbose=1, cache_size=7000)
clf.fit(x_train, y_train)

predictions = clf.predict_proba(x_dev)

# Find best split value
max_bacc = 0
best_split = 0
for split in np.arange(0.05, 0.5, 0.001):
    y_pred = [prob > split for prob in predictions[:, 1]]
    bacc = balanced_accuracy_score(y_dev, y_pred)
    if bacc > max_bacc:
        max_bacc = bacc
        best_split = split

predictions = clf.predict_proba(x_test)

# Print results
y_pred = [prob > best_split for prob in predictions[:, 1]]
print('Best split:', best_split)
print('Balanced accuracy:', balanced_accuracy_score(y_test, y_pred))
print('F1 Score:', f1_score(y_test, y_pred))
print('Accuracy:', accuracy_score(y_test, y_pred))
print('Precision:', precision_score(y_test, y_pred))
print('Recall:', recall_score(y_test, y_pred))
