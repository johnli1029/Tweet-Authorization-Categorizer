import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import math
import re
import nltk as nlp
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

# IMPORTANT: Tweets Preprocessing
def pre_processing(t):
    HTTP_URL_PATTERN = r'((http|ftp|https):\/\/)?([\w-]+(?:(?:\.[\w-]{2,})+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?'
    t = re.sub(HTTP_URL_PATTERN, '', t)
    t = re.sub('\s\W', ' ', t)
    t = re.sub('\W,\s', ' ', t)
    t = re.sub(r'\W', ' ', t)
    t = re.sub("\d+", " ", t)
    t = re.sub('\s+', ' ', t)
    t = re.sub('[!@#$_]', ' ', t)
    t = t.lower()

    lemma = nlp.WordNetLemmatizer()
    stemmed_tweet = ""
    for word in t.split():
        stemmed_tweet += lemma.lemmatize(word)
        stemmed_tweet += " "
    return stemmed_tweet.strip()

if __name__ == '__main__':
    f = open('output.log', 'w', buffering=1)
    TRAINSET_PATH = './modified.csv'
    df = pd.read_csv(TRAINSET_PATH, encoding="utf-8")
    f.write("Data set shape: ")
    f.write(str(df.shape))
    f.write('\n')

    # Cast the type of User_id field from int to string
    df.user_id = df.user_id.apply(str)

    # PreProcessing
    df.tweet = df.tweet.apply(pre_processing)

    # remove non important words such as 'a', 'the', 'that'
    # nlp.download("stopwords")  # stopwords = (irrelavent words)
    english_stopwords = set(stopwords.words('english'))
    
    ALL_TWEETS = df.tweet.tolist()
    # TODO: Find a way not to limit max features, or not pruning according to TF
    vectorizer = TfidfVectorizer(max_features=None, max_df=0.5, ngram_range=(
        1, 2), lowercase=False, analyzer='word', stop_words=english_stopwords)
    X = vectorizer.fit_transform(ALL_TWEETS)
    
    ### Important: Training
    x_test = error
    x = X
    y = df.user_id
    RANDOM_SEED = int.from_bytes("Group95".encode(), 'little') % (2**32 - 1)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=RANDOM_SEED)
    
    # Set training granularity
    SAMPLE_THRESHOLD = 10
    CLASS_DISTRO = y_train.value_counts()
    LABELS = CLASS_DISTRO[CLASS_DISTRO > SAMPLE_THRESHOLD].index
    SAMPLE_AMOUNT = y_train.shape[0]
    classifier_dict = dict()

    f.write(str(len(LABELS)))
    f.write(" Classifiers to be trained\n")
    for label in LABELS:
        label = str(label)
        weight = math.sqrt(SAMPLE_AMOUNT / CLASS_DISTRO[label])
        y_train_transformed = y_train.transform(lambda x: 1 if x == label else 0)
        y_test_transformed = y_test.transform(lambda x: 1 if x == label else 0)

        clf = LogisticRegression(
            C=1, solver='lbfgs', max_iter=200, class_weight={1: weight})
        clf.fit(x_train, y_train_transformed)
        classifier_dict[label] = clf
        if (len(classifier_dict) % 10 == 0):
            f.write(str(len(classifier_dict)))
            f.write(" Classifiers Trained!\n")

    y_pred = list()
    for vector in x_test:
        prediction = '1'
        max_proba = 0
        for label, clf in classifier_dict.items():
            proba = clf.predict_proba(vector)[0][1]
            if proba > max_proba:
                max_proba = proba
                prediction = label
    
        y_pred.append(prediction)
    
    f.write(str(accuracy_score(y_test, y_pred)))
    f.close()
