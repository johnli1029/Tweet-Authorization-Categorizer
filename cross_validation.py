import re
import logging
from time import time
from pprint import pprint
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import nltk as nlp
from nltk.corpus import stopwords

# IMPORTANT: Tweets Preprocessing
def pre_processing(t):
    HTTP_URL_PATTERN = r'((http|ftp|https):\/\/)?([\w-]+(?:(?:\.[\w-]{2,})+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?'
    t = re.sub(HTTP_URL_PATTERN, '', t)
    t = re.sub('\s\W',' ',t)
    t = re.sub('\W,\s',' ',t)
    t = re.sub(r'\W', ' ', t)
    t = re.sub("\d+", " ", t)
    t = re.sub('\s+',' ',t)
    t = re.sub('[!@#$_]', ' ', t)
    t = t.lower()

    lemma = nlp.WordNetLemmatizer()
    stemmed_tweet = ""
    for word in t.split():
        stemmed_tweet += lemma.lemmatize(word)
        stemmed_tweet += " "
    return stemmed_tweet.strip()


# Author: Olivier Grisel <olivier.grisel@ensta.org>
#         Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Mathieu Blondel <mathieu@mblondel.org>
# License: BSD 3 clause


print(__doc__)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


##############################################################################
TRAINSET_PATH = './modified.csv'
df = pd.read_csv(TRAINSET_PATH, encoding="utf-8")
print("DataFrame shape:")
print(df.shape)

# Cast the type of User_id field from int to string
df.user_id = df.user_id.apply(str)
# Pre-processing on tweets
df.tweet = df.tweet.apply(pre_processing)

# Set X and Y 
X = df.tweet.tolist()
Y = df.user_id

Y_transformed = Y.transform(lambda x: 1 if x == '4185' else 0)
print("%d tweets" % len(X))
print("%d categories" % len(Y_transformed.value_counts()))
print()

# #############################################################################
# Define a pipeline combining a text feature extractor with a simple
# classifier
english_stopwords = set(stopwords.words('english'))
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(lowercase=False,
                              analyzer='word', stop_words=english_stopwords)),
    ('clf', LogisticRegression(solver='lbfgs')),
])

# uncommenting more parameters will give better exploring power but will
# increase processing time in a combinatorial way
parameters = {
    'tfidf__max_df': (0.5, 0.75, 1.0),
    'tfidf__max_features': (None, 5000, 10000),
    # 'tfidf__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    #     'tfidf__use_idf': (True, False),
    # 'tfidf__norm': ('l1', 'l2'),
    'clf__max_iter': (200,),
    'clf__C': (0.01, 0.05, 0.1, 0.3, 0.5, 1.0),
    #     'clf__penalty': ('l2', 'l1'),
    'clf__class_weight': ({1: 30}, {1: 50}, {1: 100})
}

if __name__ == "__main__":
    # multiprocessing requires the fork to happen in a __main__ protected
    # block

    # find the best parameters for both the feature extraction and the
    # classifier
    grid_search = GridSearchCV(pipeline, parameters, scoring='f1', cv=3,
                               n_jobs=-1, verbose=1)

    print("Performing grid search...")
    print("pipeline:", [name for name, _ in pipeline.steps])
    print("parameters:")
    pprint(parameters)
    t0 = time()
    grid_search.fit(X, Y_transformed)
    print("done in %0.3fs" % (time() - t0))
    print()

    print("Best score: %0.3f" % grid_search.best_score_)
    print("Best parameters set:")
    best_parameters = grid_search.best_estimator_.get_params()
    for param_name in sorted(parameters.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))

