# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 15:02:22 2016

@author: mattia

Let's see what happens when the classifier is trained in a dataset
and it's tested in the other one
"""
import pathlib as pl
import pandas as pd
import numpy as np
import copy
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from xgboost import XGBClassifier
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
import sys
import os
import logging
from wallace_db import WallaceDBHelper
lib_path = os.path.abspath(os.path.join('..', 'models'))
sys.path.append(lib_path)
sys.path.append(os.path.abspath(os.path.join('..')))
from models.srSVD import SquareRootSVD
from models.classicSVD import ClassicLSA


def iscapslock(doc):
    """Returns true whether at least one word is majuscule"""
    for word in wordpunct_tokenize(doc):
        if word.isupper():
            return True
    return False
    
def multiple(char, doc):
    """Returns true whether a character is repeated multiple times in a row"""
    docp = doc
    pos = docp.find(char)
    while pos < len(docp) and pos > -1:
        nextpos = docp[pos:].find(char)
        if nextpos == 0:
            return True
        elif nextpos == -1:
            return False
        else:
            pos = nextpos
            docp = doc[pos:]
    return False
            
def ispresent(doc, term):
    return doc.find(term) >= 0
    
    
if __name__ == '__main__':
    
    

    program = os.path.basename(sys.argv[0])
    
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info('running %s' % ' '.join(sys.argv))
    
    
    """Select dataset here"""
    df1 = pd.read_csv('../datasets/filatova.csv', header=0, index_col=0)
    df2 = pd.read_csv('../datasets/data-sarc-sample.csv', header=0, index_col=0)
    
    """Extracting features of interest"""
    labels1 = np.array(df1['label'])
    reviews1 = np.array(df1['text'])

    texts2 = np.array(df2['text'])
    labels2 = np.array(df2['label'])
    
    dbHelper = WallaceDBHelper()
    reviews, labels = dbHelper.get_texts_and_labels()
    reviews3 = np.array(reviews)
    labels3 = np.array(labels)

    df4 = pd.read_csv('../datasets/sarcasmv2.csv', header=0, index_col=0)
    texts4 = df4['Response Text']
    labels4 = np.array(df4["Label"] == "sarc").astype(int)
    
    del df1, df2, df4
    
    classifiers = [SVC(C=100, kernel="linear", class_weight='balanced'), \
          SVC(C=100, kernel='rbf', class_weight='balanced'), \
            LogisticRegression(class_weight='balanced', penalty='l1', C=10), \
            LogisticRegression(class_weight='balanced', penalty='l2', C=10), \
            RandomForestClassifier(n_estimators=300, criterion='gini'), \
            RandomForestClassifier(n_estimators=300, criterion='entropy'), \
            DecisionTreeClassifier(criterion='gini'), \
            DecisionTreeClassifier(criterion='entropy'), \
            GaussianNB(), \
            XGBClassifier(max_depth=5, n_estimators=500)]
    
    names = ['svm linear', 'svm gaussian', 'lr_L1', 'lr_L2', 'RF_gini', \
        'RF_entropy', 'DT_gini', 'DT_entropy', 'Bayes', 'XGB']
            
    datasets = ['SarcasmCorpus', 'data-sarc-sample', 'wallace', 'sarcasmv2']
    
    dimensions = [40, 90, 140, 200, 250]
    
    extra_features = False
    
     
    i = 0
    """ Performing cross validation """
    train_texts1 = reviews1
    train_texts2 = texts2
    train_texts3 = reviews3
    train_texts4 = texts4
    
    res_path = '../results/cross/'
    if not pl.Path(res_path).exists():
        os.mkdir(res_path)
    
    lsa = None
    tokenizer = wordpunct_tokenize
    if sys.argv[1] == 'classic':
        vectorizer = TfidfVectorizer(ngram_range=(1, 1), stop_words=stopwords.words('english'), tokenizer=tokenizer)
        lsa = ClassicLSA
        res_path += '/classic/'
        if not pl.Path(res_path).exists():
            os.mkdir(res_path)
    else:
        vectorizer = CountVectorizer(ngram_range=(1, 1), stop_words=stopwords.words('english'),
                                     tokenizer=tokenizer)
        lsa = SquareRootSVD

    i, j = 0, 0
    for train_texts, labels in zip([train_texts1, train_texts2, train_texts3, train_texts4], [labels1, labels2, labels3, labels4]):
        for test_texts, test_labels in zip([train_texts1, train_texts2, train_texts3, train_texts4], [labels1, labels2, labels3, labels4]):
            if i == j:
                j += 1
                continue

            for k in dimensions:
                print('k = ' + str(k))
                expdir = pl.Path(res_path + str(k) + '/')
                if not expdir.exists():
                    os.mkdir(str(expdir))

                precisions = np.zeros(len(names),)
                recalls = np.zeros(len(names),)
                f_scores = np.zeros(len(names),)
                accuracies = np.zeros(len(names),)

                models = list()
                models.append(lsa(vectorizer=vectorizer, classifier=classifiers[0], k=k))

                models[0].fit(train_texts.T, labels)

                """For each model, train it and test"""
                for l in range(1, len(classifiers)):
                    model = classifiers[l]
                    models.append(copy.copy(models[0]))
                    models[l].set_params(**{"classifier": model})

                for l in range(len(models)):
                    out = models[l].predict(test_texts)
                    precisions[l] = (precision_score(test_labels, out))
                    recalls[l] = (recall_score(test_labels, out))
                    f_scores[l] = (f1_score(test_labels, out))
                    accuracies[l] = (accuracy_score(test_labels, out))
                df_out = pd.DataFrame({
                    'precision': precisions,
                    'recall': recalls,
                    'f_score': f_scores,
                    'accuracy': accuracies,
                    'model': names
                })
                df_out.to_csv("{0}/{1}-->{2}.csv".format(expdir, datasets[i], datasets[j]), index=False)
            j += 1
        i += 1
        j = 0
