# -*- coding: utf-8 -*-
import pandas as pd
from urllib.parse import urlparse
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
# using 8 general lexical features which may or may not affect URL status
# features which need to be generated are:
# URL length
# Presence of HTTPS in URL
# number of HTTP in URL
# length of domain
# number of digits in the URL
# number of @s in url
# number of '.' in url
# number of sub-directories in URL path

TYPES = ['benign', 'defacement', 'malware', 'phishing']

def contains_https(url):
    if 'https' in url:
        return 1
    else:
        return 0

def get_domain_length(url):
    return len(urlparse(str(url)).netloc)

def get_num_of_digits(url):
    num = 0
    for i in url:
        if i.isnumeric():
            num = num + 1
    return num

def count_sub_dir(url):
    directory = urlparse(url).path
    return directory.count('/')

def genearte_features(file_path):

    print("[INFO] importing data...\n")
    data = pd.read_csv(file_path)

    print("[INFO] Generating features...\n")
    # URL length
    data['length'] = data['url'].apply(lambda i: len(str(i)))
    # presence of https in URL
    data['https'] = data['url'].apply(lambda i: contains_https(i))
    # number of http in URL
    data['http'] = data['url'].apply(lambda i: i.count('http'))
    # length of domain
    # this does not seem to work for URLs which do not have a leading protocol scheme 
    # i.e. 'google.com' vs 'https://google.com'
    # as such, this feature may be excluded as the effect on the model is questionable
    data['domain_length'] = data['url'].apply(lambda i: get_domain_length(i))
    # number of digits in domain
    data['digits'] = data['url'].apply(lambda i: get_num_of_digits(i))
    # number of @s present in URL
    data['ats'] = data['url'].apply(lambda i: i.count('@'))
    # number of . in url
    data['dots'] = data['url'].apply(lambda i: i.count('.'))
    # number of sub-directories
    data['directories'] = data['url'].apply(lambda i: count_sub_dir(i))

    return data

def input_fn(data, batch_size):
    features = ['length', 'https', 'http', 'domain_length', 'digits', 'ats', 'dots', 'directories']

    data['type'] = pd.Categorical(data['type'])
    data['type'] = data.type.cat.codes

    target = data.pop('type')

    dataset = tf.data.Dataset.from_tensor_slices((data[features].to_dict('list'), tf.cast(target.values, tf.int32)))
    #for feat, targ in dataset.take(500):
    #    print ('Features: {}, Target: {}'.format(feat, targ))
    dataset = dataset.shuffle(len(data)).repeat().batch(batch_size)
    return dataset

length = tf.feature_column.numeric_column('length')
https = tf.feature_column.indicator_column(tf.feature_column.categorical_column_with_identity('https', 2))
http = tf.feature_column.numeric_column('http')
domain_length = tf.feature_column.numeric_column('domain_length')
digits = tf.feature_column.numeric_column('digits')
ats = tf.feature_column.numeric_column('ats')
dots = tf.feature_column.numeric_column('dots')
directories = tf.feature_column.numeric_column('directories')

#fetch the malicious_phish.csv dataset from drive.google.com
shr_url='https://drive.google.com/file/d/11aF3BWiQLhMSRAB6-UbKJap-OopCDa_E/view?usp=sharing'
dwn_url = 'https://drive.google.com/uc?id=' + shr_url.split('/')[-2]
feature_columns = [length, https, http, domain_length, digits, ats, dots, directories]
dataframe = genearte_features(dwn_url)

model = tf.estimator.DNNClassifier(feature_columns=feature_columns, hidden_units=[30,60,20], n_classes=4, model_dir="./model")
batch_size = 100

print("\n[INFO] training model...\n")
model.train(steps=1000, input_fn=lambda : input_fn(dataframe, batch_size))

def get_features(url):
    features = {'length': [len(str(url))], 
                'https': [contains_https(url)], 
                'http': [url.count('http')], 
                'domain_length': [get_domain_length(url)], 
                'digits': [get_num_of_digits(url)], 
                'ats': [url.count('@')], 
                'dots': [url.count('.')], 
                'directories': [count_sub_dir(url)] }
    return features

def pred_input_fn(features, batch_size=256):
    return tf.data.Dataset.from_tensor_slices(features).batch(batch_size)

url = None

while url != 'd':
    print("\nEnter a URL for prediction or enter 'd' to quit: ")
    url = input()
    if url != 'd':
        features = get_features(url)

        pred = model.predict(input_fn=lambda : pred_input_fn(features))

        expected = [0]

        for p_dict in pred:
            class_id = p_dict['class_ids'][0]
            probability = p_dict['probabilities'][class_id]
            print('\nprediction is "{}" with a probability of ({:.1f}%)\n'.format(TYPES[class_id], probability))