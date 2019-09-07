import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import math
from feature_engineering import pre_processing
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, HashingVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from multiprocessing import Pool

if __name__ == '__main__':
    # Read Data
    TRAINSET_PATH = './modified.csv'
    TEST_SET_PATH = './test_tweets_unlabeled.txt'
    df = pd.read_csv(TRAINSET_PATH, encoding="utf-8")
    x_test_df = pd.read_csv(TEST_SET_PATH, sep='\n',
                            encoding="utf-8", quoting=3, names=["tweet"])
    print(df.shape)
    print(x_test_df.shape)

    # feature enginerring
    df.user_id = df.user_id.apply(str)
    PROCESS_AMOUNT = 10
    with Pool(processes=PROCESS_AMOUNT) as pool:
        df.tweet = pool.map(pre_processing, df.tweet)
        x_test_df.tweet = pool.map(pre_processing, x_test_df.tweet)
    print("Tweet Preprocessing Done!")

    ALL_TWEETS = df.tweet.tolist() + x_test_df.tweet.tolist()
    english_stopwords = set(stopwords.words('english'))
    # TODO: Find a way not to limit max features, or not pruning according to TF
    # vectorizer = TfidfVectorizer(max_features=50000, max_df=0.75, ngram_range=(
    #     1, 2), lowercase=False, analyzer='word', stop_words=english_stopwords)
    
    # X = vectorizer.fit_transform(ALL_TWEETS)
    # print("TF-IDF Tokenizer Done!")
    
    vectorizer = HashingVectorizer(decode_error='ignore', n_features=2**15, alternate_sign=False,
                                   lowercase=False, analyzer='word', stop_words=english_stopwords)

    ### Important: Training
    # x_train = X[0:df.tweet.shape[0]]
    # x_pred = X[df.tweet.shape[0]:]
    # y_train = df.user_id


    batch_size = 170000
    clf = ComplementNB()
    all_classes = df.user_id.value_counts().index
    iteration = df.shape[0] // batch_size + 1
    for i in range(iteration):
        x_train_text = df.tweet[i*batch_size:(i+1)*batch_size]
        x_train = vectorizer.transform(x_train_text)
        y_train = df.user_id[i*batch_size:(i+1)*batch_size]
        clf.partial_fit(x_train, y_train, classes=all_classes)
    print("NB Fitting Done!")

    y_pred = list()
    iteration = x_test_df.shape[0] // batch_size + 1
    for i in range(iteration):
        x_test_text = x_test_df.tweet[i*batch_size:(i+1)*batch_size]
        x_test = vectorizer.transform(x_test_text)
        y_pred += clf.predict(x_test).tolist()

    output = pd.DataFrame({'Id': range(1, x_test_df.shape[0] + 1),
                           'Predicted': y_pred})

    output.to_csv('./submission.csv', index=False)
