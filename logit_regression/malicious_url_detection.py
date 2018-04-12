import re
import numpy as np
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def get_tokenizers(url):
    tokenizers = []
    split_by_slash = url.split('/')
    for data in split_by_slash:
        term = re.split(r'[.-]', data)
        tokenizers = tokenizers + term
    tokenizers = list(set(tokenizers))

    if 'com' in tokenizers:
        tokenizers.remove('com')
    return tokenizers


def get_urls(url_path):
    # url_path = '/Users/guoyifeng/PycharmProjects/MaliciousURLDetection/logit_regression/urls.csv'

    urls = []

    with open(url_path) as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        for row in readCSV:
            urls.append(row)

    urls = np.array(urls)
    np.random.shuffle(urls)
    return urls


def train(dataset):
    urls = []
    labels = []
    for data in dataset:
        urls.append(data[0])
        labels.append(data[1])

    vectorizer = TfidfVectorizer(tokenizer=get_tokenizers)

    X = vectorizer.fit_transform(urls)

    X_train, X_test, labels_train, labels_test = train_test_split(X, labels, test_size=0.2, random_state=42)

    lgs = LogisticRegression()
    lgs.fit(X_train, labels_train)
    print(lgs.score(X_test, labels_test))
    return vectorizer, lgs


if __name__ == '__main__':
    url_path = '/Users/guoyifeng/PycharmProjects/MaliciousURLDetection/logit_regression/urls.csv'
    dataset = get_urls(url_path)

    vectorizer, lgs = train(dataset)

    X_predict = ['wikipedia.com','google.com/search=faizanahad','pakistanifacebookforever.com/getpassword.php/','www.radsport-voggel.de/wp-admin/includes/log.exe','ahrenhei.without-transfer.ru/nethost.exe','www.itidea.it/centroesteticosothys/img/_notes/gum.exe']

    X_predict = vectorizer.transform(X_predict)

    y_Predict = lgs.predict(X_predict)

    print(y_Predict)	#printing predicted values


