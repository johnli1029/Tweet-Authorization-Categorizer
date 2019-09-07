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
from multiprocessing import Pool

# IMPORTANT: Tweets Preprocessing


def pre_processing(t):
    HTTP_URL_PATTERN = r'((http|ftp|https):\/\/)?([\w-]+(?:(?:\.[\w-]{2,})+))([\w.,@?^=%&:/~+#-]*[\w@?^=%&/~+#-])?'
    t = t.strip()
    t = re.sub(HTTP_URL_PATTERN, ' HAVELINK ', t)  # Increase URL weight
    t = re.sub(r'@handle', ' ATSOMEBODY ', t)  # Deal with @handle
    t = re.sub(r'^RT', ' APURERETWEET ', t)
    t = re.sub(r'RT(?!\w)', ' WITHARETWEET ', t)

    # Find hashtag and emphasize it
    m = re.search(r'#(?P<hashtag>\w+?)\b', t)
    while m is not None:
        hashtag = m.group('hashtag')
        t = re.sub(r'#'+hashtag, (' '+hashtag+' ')*3, t)
        m = re.search(r'#(?P<hashtag>\w+?)\b', t)

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
    TRAINSET_PATH = './modified.csv'
    TEST_SET_PATH = './test_tweets_unlabeled.txt'
    df = pd.read_csv(TRAINSET_PATH, encoding="utf-8")
    x_test_df = pd.read_csv(TEST_SET_PATH, sep='\n',
                            encoding="utf-8", quoting=3, names=["tweet"])
    print(df.shape)
    print(x_test_df.shape)

    df.user_id = df.user_id.apply(str)
    y_train = df.user_id
    #    SAMPLE_THRESHOLD = 20
    #    CLASS_DISTRO = y_train.value_counts()
    #    LABELS = CLASS_DISTRO[CLASS_DISTRO > SAMPLE_THRESHOLD].index
    #    print("{} Classifiers to be trained".format(len(LABELS)))

    df.tweet = df.tweet.apply(pre_processing)
    x_test_df.tweet = x_test_df.tweet.apply(pre_processing)

    # remove non important words such as 'a', 'the', 'that'
    # nlp.download("stopwords")  # stopwords = (irrelavent words)
    english_stopwords = set(stopwords.words('english'))

    ALL_TWEETS = df.tweet.tolist() + x_test_df.tweet.tolist()
    # TODO: Find a way not to limit max features, or not pruning according to TF
    vectorizer = TfidfVectorizer(max_features=10000, max_df=0.75, ngram_range=(
        1, 2), lowercase=False, analyzer='word', stop_words=english_stopwords)
    X = vectorizer.fit_transform(ALL_TWEETS)

    ### Important: Training
    x_train = X[0:df.tweet.shape[0]]
    x_pred = X[df.tweet.shape[0]:]
    y_train = df.user_id

    # from sklearn.linear_model import LogisticRegression
    clf = LogisticRegression(
        C=1, solver='lbfgs', max_iter=1000, class_weight='balanced', multi_class='multinomial')
    clf.fit(x_train, y_train)

    # from sklearn.metrics import accuracy_score
    y_pred = clf.predict(x_pred)

    output = pd.DataFrame({'Id': range(1, x_pred.shape[0] + 1),
                           'Predicted': y_pred})

    output.to_csv('./submission.csv', index=False)
