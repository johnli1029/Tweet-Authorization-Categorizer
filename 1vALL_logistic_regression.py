import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import math
from feature_engineering import pre_processing
from nltk.corpus import stopwords
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from multiprocessing import Pool

def predict(vector):
    global classifier_dict
    prediction = '1'
    max_proba = -1
    for label, clf in classifier_dict.items():
        proba = clf.predict_proba(vector)[0][1]
        if proba > max_proba:
            max_proba = proba
            prediction = label

    return prediction


if __name__ == '__main__':
    TRAINSET_PATH = './modified.csv'
    TEST_SET_PATH = './test_tweets_unlabeled.txt'
    df = pd.read_csv(TRAINSET_PATH, encoding="utf-8")
    x_test_df = pd.read_csv(TEST_SET_PATH, sep='\n',
                            encoding="utf-8", quoting=3, names=["tweet"])
    print(df.shape)
    print(x_test_df.shape)


    df.user_id = df.user_id.apply(str)
    # SAMPLE_THRESHOLD = 20
    # CLASS_DISTRO = y_train.value_counts()
    # LABELS = CLASS_DISTRO[CLASS_DISTRO > SAMPLE_THRESHOLD].index
    # print("{} Classifiers to be trained".format(len(LABELS)))
    
    PROCESS_AMOUNT = 10
    with Pool(processes=PROCESS_AMOUNT) as pool:
        df.tweet = pool.map(pre_processing, df.tweet)
        x_test_df.tweet = pool.map(pre_processing, x_test_df.tweet)
    print("Tweet Preprocessing Done!")

    # remove non important words such as 'a', 'the', 'that'
    # nlp.download("stopwords")  # stopwords = (irrelavent words)
    english_stopwords = set(stopwords.words('english'))
    
    ALL_TWEETS = df.tweet.tolist() + x_test_df.tweet.tolist()
    # TODO: Find a way not to limit max features, or not pruning according to TF
    vectorizer = TfidfVectorizer(max_features=None, max_df=0.75, ngram_range=(
        1, 2), lowercase=False, analyzer='word', stop_words=english_stopwords)
    X = vectorizer.fit_transform(ALL_TWEETS)
    
    ### Important: Training
    x_train = X[0:df.tweet.shape[0]]
    x_pred = X[df.tweet.shape[0]:]
    y_train = df.user_id

    # Set training granularity
    SAMPLE_THRESHOLD = 20
    CLASS_DISTRO = y_train.value_counts()
    LABELS = CLASS_DISTRO[CLASS_DISTRO > SAMPLE_THRESHOLD].index
    SAMPLE_AMOUNT = y_train.shape[0]
    classifier_dict = dict()

    print("{} Classifiers to be trained".format(len(LABELS)))
    for label in LABELS:
        label = str(label)
        weight = math.sqrt(SAMPLE_AMOUNT / CLASS_DISTRO[label])
        y_train_transformed = y_train.transform(lambda x: 1 if x == label else 0)

        clf = LogisticRegression(
            C=1, solver='lbfgs', max_iter=200, class_weight={1: weight})
        clf.fit(x_train, y_train_transformed)
        classifier_dict[label] = clf
        if (len(classifier_dict) % 10 == 0):
            print("{} Classifiers Trained!".format(len(classifier_dict)))

    del x_train, y_train
    y_pred = list()
    for vector in x_pred:
        y_pred.append(predict(vector))

    output = pd.DataFrame({'Id': range(1, x_pred.shape[0] + 1),
                           'Predicted': y_pred})

    output.to_csv('./submission.csv', index=False)
