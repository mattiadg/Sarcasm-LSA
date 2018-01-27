# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 16:51:06 2016

@author: mattia
"""
import pathlib as pl
import argparse
import os
import sys
import copy
from argparse import _ActionsContainer

lib_path = os.path.abspath(os.path.join('..', 'models'))
sys.path.append(lib_path)
sys.path.append(os.path.abspath(os.path.join('..')))

from models.srSVD import SquareRootSVD
from models.classicSVD import ClassicLSA

import pandas as pd
import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from nltk import pos_tag
from wallace_db import WallaceDBHelper
import re


def get_terms(doc, adj_and_adv=False):
    pattern = '[a-z][a-z]+'
    tokens = wordpunct_tokenize(doc)
    if adj_and_adv:
        tokens = pos_tag(tokens)
        tokens = [token[0] for token in tokens if token[1] in ['RB', 'JJ']]
    return filter(lambda token: re.search(pattern, token), tokens)


def preprocess(doc, _stopwords, _tokenizer):
    return np.array([" ".join([word.lower() for word in _tokenizer(d) if word not in _stopwords]) for d in doc])


def read_dataset(dsname):
    dsdir = '../datasets/'
    index_col = 0
    if dsname == 'wallace':
        text, label = WallaceDBHelper().get_texts_and_labels()
        return text, label, None

    df = pd.read_csv(dsdir + dsname + ".csv", header=0, index_col=index_col)
    more = None
    if dsname in ['filatova', 'data-sarc-sample']:
        text = df['text']
        label = np.array(df['label']).astype(int)

        if dsname == "filatova":
            more = df['stars']

    elif dsname.startswith('sarcasmv2'):
        text = df['Response Text']
        label = np.array(df["Label"] == "sarc").astype(int)

    return text, label, more
        
tokenizer = wordpunct_tokenize
    
names = ['svm linear', 'svm gaussian', 'lr_L1', 'lr_L2',
         'RF_entropy', 'DT_entropy', 'Bayes', 'XGB']

if __name__ == '__main__':
    file_path = '../results/'

    # Arguments parsing
    parser = argparse.ArgumentParser()

    parser.add_argument("-e", "--experiment", help="the name of current experiment", default="grid_search", dest="exp")
    parser.add_argument("--n-fold", help="The number of folds for cross-validation", type=int, default=10, dest="nfold")
    parser.add_argument("--n-run", help="The number of runs of cross-validation", type=int, default=1, dest="nrun")
    parser.add_argument("-d", "-dataset",
                        help="The name of the data set to experiment on. Valid values are: filatova, data-sarc-sample, \
                        wallace and all",
                        type=str, default="all", dest="ds")
    parser.add_argument("--stars", help="With data set Filatova, add the number of stars as a feature. \
                        It's useless for the others",
                        default=False, action='store_const', const=True, dest="star")
    parser.add_argument("--classic", help="Use classic LSA instead of sqrt" , \
                         action='store_const', const="classic", default="sqrt", dest="svd")

    args = parser.parse_args()

    dir_pt = pl.Path(file_path)
    if not dir_pt.exists():
        os.mkdir(file_path)
    if not dir_pt.joinpath(args.exp).exists():
        os.mkdir(file_path + args.exp)

    all_datasets = ['filatova', 'data-sarc-sample', 'wallace', 'sarcasmv2', 'sarcasmv2_gen', 'sarcasmv2_hyp', 'sarcasmv2_rq']

    if args.ds == 'all':
        datasets = all_datasets
    elif args.ds not in all_datasets:
        raise AttributeError
    else:
        datasets = [args.ds]

    vectorizer = None
    lsa = None
    if args.svd == "sqrt":
        vectorizer = CountVectorizer(ngram_range=(1, 1), stop_words=stopwords.words('english'),
                                     tokenizer=tokenizer)
        lsa = SquareRootSVD
    elif args.svd == "classic":
        vectorizer = TfidfVectorizer(ngram_range=(1, 1), stop_words=stopwords.words('english'), tokenizer=tokenizer)
        lsa = ClassicLSA
    else:
        raise NotImplementedError
    for l in range(len(datasets)):
        
        Rs = [20, 40, 90, 120, 140, 170, 200, 220, 250, 280, 300, 350]
        # Mean scores in N run
        n_run = args.nrun
        n_folds = args.nfold

        chosen_ds = datasets[l]

        texts, labels, more = read_dataset(chosen_ds)

        print("Dataset: " + chosen_ds)
            
        for u in range(len(Rs)):
            k = Rs[u]
            print("k=" + str(k))
            for n in range(n_run):
                classifiers = [SVC(C=100, kernel="linear", class_weight='balanced'),
                               SVC(C=100, kernel='rbf', class_weight='balanced'),
                               LogisticRegression(class_weight='balanced', penalty='l1', C=10),
                               LogisticRegression(class_weight='balanced', penalty='l2', C=10),
                               RandomForestClassifier(n_estimators=150, criterion='entropy'),
                               DecisionTreeClassifier(criterion='entropy'),
                               GaussianNB(),
                               XGBClassifier(max_depth=5, n_estimators=500)]
                # Mean scores in K-FOLD
                precisions = np.zeros((n_folds, len(classifiers)))
                recalls = np.zeros((n_folds, len(classifiers)))
                f_scores = np.zeros((n_folds, len(classifiers)))
                accuracies = np.zeros((n_folds, len(classifiers)))
                cv_s = np.zeros((n_folds*len(classifiers)))

                i = 0
                """ Performing cross validation """
                # print('Begin k-fold')
                for train_idx, test_idx in StratifiedKFold(labels, n_folds=n_folds, shuffle=True):
                    print("FOLD: " + str(i+1))

                    X_train = texts[train_idx]
                    y_train = labels[train_idx]
                    X_test = texts[test_idx]
                    y_test = labels[test_idx]

                    models = list()
                    models.append(lsa(vectorizer=vectorizer, classifier=classifiers[0], k=k))

                    if chosen_ds == 'filatova' and args.star:
                        models[0].fit(X_train.T, y_train, more=more[train_idx])
                    else:
                        models[0].fit(X_train.T, y_train)
                    """For each model, train it and test"""
                    for j in range(1, len(classifiers)):
                        model = classifiers[j]
                        models.append(copy.copy(models[0]))
                        models[j].set_params(**{"classifier": model})

                    for j in range(len(models)):
                        print("Model: " + names[j])
                        if chosen_ds == 'filatova' and args.star:
                            out = models[j].predict(X_test, more=more[test_idx])
                        else:
                            out = models[j].predict(X_test)
                        precisions[i, j] = precision_score(y_test, out)
                        recalls[i, j] = recall_score(y_test, out)
                        f_scores[i, j] = f1_score(y_test, out, average='binary')
                        accuracies[i, j] = accuracy_score(y_test, out)
                        start = len(models)*i
                        cv_s[start:start+len(models)] = i

                    i += 1

                """Save data in DataFrame and store it"""
                v_len = n_folds*len(models)
                rep_names = np.tile(names, n_folds)
                df_out = pd.DataFrame({
                    'precision' : np.reshape(precisions, (v_len,)),
                    'recall'    : np.reshape(recalls, (v_len,)),
                    'f_score'   : np.reshape(f_scores, (v_len,)),
                    'accuracy'  : np.reshape(accuracies, (v_len,)),
                    'cv'        : np.reshape(cv_s, (v_len,)),
                    'model'     : rep_names
                })

                """Store"""
                dir_path = '../results/' + args.exp

                dir_path += '/' + chosen_ds + '/'
                path = pl.Path(dir_path)
                if not path.exists():
                    os.mkdir(dir_path)
                out_file = dir_path + 'run' + str(n+1) + 'k' + str(k) + '.csv'
                if chosen_ds == "filatova" and args.star:
                    out_file = out_file[:-4] + "_stars.csv"
                df_out.to_csv(out_file)
