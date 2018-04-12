import pandas as pd
import numpy as np
import csv

url_path = '/Users/guoyifeng/PycharmProjects/MaliciousURLDetection/logit_regression/urls.csv'

urls = []

with open(url_path) as csvfile:
    readCSV = csv.reader(csvfile, delimiter=',')
    for row in readCSV:
        urls.append(row)

urls = np.array(urls)
np.random.shuffle(urls)

for url in urls:
    print(url[0] + " " +url[1])
