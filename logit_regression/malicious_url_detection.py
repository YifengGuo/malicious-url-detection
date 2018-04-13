import re
import numpy as np
import csv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
from logit_regression.LogitRegression import LogitRegression


# def get_tokenizers(url):
#     tokenizers = []
#     split_by_slash = url.split('/')
#     for data in split_by_slash:
#         term = re.split(r'[.-]', data)
#         tokenizers = tokenizers + term
#     tokenizers = list(set(tokenizers))
#
#     if 'com' in tokenizers:
#         tokenizers.remove('com')
#     return tokenizers


def get_tokenizers(url):
    # www.google.com/23241312/search-path-s111/2314
    tokenizers = []
    split_by_slash = str(url).split('/')
    for part in split_by_slash:
        split_by_dot = []
        data = str(part).split('-')
        for i in range(len(data)):
            min_part = str(data[i]).split('.')
            split_by_dot = split_by_dot + min_part
            tokenizers = data + split_by_dot + tokenizers
    tokenizers = list(set(tokenizers)) # deduplicate terms in tokenizers
    if 'com' in tokenizers:
        tokenizers.remove('com')
    if '' in tokenizers:
        tokenizers.remove('')
    tokenizers = [x for x in tokenizers if '.' not in x]
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

    # print (vectorizer)

    X = vectorizer.fit_transform(urls)

    # print(X)

    X_train, X_test, labels_train, labels_test = train_test_split(X, labels, test_size=0.2)

    # print(X_train.toarray())

    # print(X_train)
    # print(type(X_train[0]))
    # print(len(labels_train))

    # lgs = LogisticRegression()
    # lgs.fit(X_train, labels_train)
    # print(lgs.score(X_test, labels_test))
    # return vectorizer, lgs

    # labels_train = np.array(labels_train)
    labels_test = np.array(labels_test)
    # X_train = np.array(X_train)
    # X_test = np.array(X_test)

    X_train_array = X_train.toarray()

    lgs = LogitRegression()
    # weights = lgs.train_logit_regression(X_train.toarray(), labels_train.T, 1, 0.01)
    for i in range(len(X_train_array)):
        print(X_train_array[i])
        weights = lgs.train_logit_regression(X_train_array[i], labels_train[i], 1, 0.01)
    print(weights)
    # X_train_array = X_train.toarray()
    # labels_train_array =
    accuracy = lgs.test_accuarcy(X_test.toarray(), labels_test.T, weights)

    return accuracy



if __name__ == '__main__':
    url_path = '/Users/guoyifeng/PycharmProjects/MaliciousURLDetection/logit_regression/urls.csv'
    dataset = get_urls(url_path)

    # vectorizer, lgs = train(dataset)

    train(dataset)
    # accuracy = train(dataset)

    # print(accuracy)

    X_predict = ['wikipedia.com','google.com/search=faizanahad','pakistanifacebookforever.com/getpassword.php/','www.radsport-voggel.de/wp-admin/includes/log.exe','ahrenhei.without-transfer.ru/nethost.exe']

    # X_predict = ['http://www.paypal.com.serv-5-redirect.aortadobrasil.com/cgi-bin/us/webscr.php?cmd=_login-run']

    # for i in range(len(X_predict)):
    #     print(get_tokenizers(X_predict[i]))

    # X_predict = vectorizer.transform(X_predict)
    #
    # y_Predict = lgs.predict(X_predict)
    #
    # X_predict = np.array(X_predict)
    # print(type(X_predict))

    # print(y_Predict)	#printing predicted values

    # print(get_tokenizers(X_predict[0]))
    # vectorizer2 = TfidfVectorizer(tokenizer=get_tokenizers)
    # X_predict = vectorizer2.fit_transform(X_predict)

    # print(X_predict)

    #print(X_predict.toarray())



