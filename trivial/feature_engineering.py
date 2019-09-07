import re
import nltk as nlp

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
