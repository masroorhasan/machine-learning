# imports
import pandas as pd

# Dataset imported from https://archive.ics.uci.edu/ml/machine-learning-databases/00228/
df = pd.read_table('smsspamcollection/SMSSpamCollection',
                    sep='\t',
                    header=None,
                    names=['label', 'sms_message'])

# print first 5 columns
print df.head()

'''
    data pre-processing
    use binary labels for spam vs not spam
    ham (not spam): 0
    spam: 1
'''
df['label'] = df.label.map({'ham':0, 'spam':1})
print df.shape  # num of rows and cols
print df.head()


# split into training and testing sets
from sklearn.cross_validation import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df['sms_message'],
                                                    df['label'],
                                                    random_state=1)

print('Number of rows in the total set: {}'.format(df.shape[0]))
print('Number of rows in the training set: {}'.format(x_train.shape[0]))
print('Number of rows in the test set: {}'.format(x_test.shape[0]))

'''
    feature extraction into bag of words
'''
from sklearn.feature_extraction.text import CountVectorizer
count_vector = CountVectorizer()

# fit training data and return matrix
training_data = count_vector.fit_transform(x_train)

# transform testing data and return matrix
testing_data = count_vector.transform(x_test)

'''
Use naive bayes to train
'''
from sklearn.naive_bayes import MultinomialNB
nb = MultinomialNB()
nb.fit(training_data, y_train)

# predict
pred = nb.predict(testing_data)

'''
Accuracy
'''
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print('Accuracy score: ', format(accuracy_score(y_test, pred)))
print('Precision score: ', format(precision_score(y_test, pred)))
print('Recall score: ', format(recall_score(y_test, pred)))
print('F1 score: ', format(f1_score(y_test, pred)))

