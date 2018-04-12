import re


def parseURL(url):
    features = []
    split_by_slash = url.split('/')
    for data in split_by_slash:
        term = re.split(r'[.-]', data)
        features = features + term
    features = list(set(features))

    if 'com' in features:
        features.remove('com')
    return features





if __name__ == "__main__":
    url = "www.baidu.com/jdisjdiw/123"
    features = parseURL(url)
    print(features)