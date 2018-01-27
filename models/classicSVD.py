# -*- coding: utf-8 -*-
"""
Created on Mon Feb 29 10:51:43 2016

@author: mattiadigangi
"""
import numpy as np
from scipy.sparse.linalg import svds


class ClassicLSA:
    """This class creates a model that takes as input a text corpus,
        apply a vectorization algorithm to it, then computes TSVD,  andIn the end
        it applies a classifier to the vectors produced this way.
        Both the vectorizer and the classifier are input from the outside by using the
        Strategy design pattern.
    """

    def __init__(self, vectorizer, classifier, k=100):
        """
        :param vectorizer: transform a text corpus in a matrix consisting of fixed-size document vectors
        :param classifier: A supervised learning algorithm for classification.
        :param k: the number of dimensions to keep for TSVD
        :type vectorizer: sklearn.Transformer it must be fit to a data set and implement the method transform for new data
        :type classifier: sklearn.Predictor it must provide class labels for data
        :type k: int
        """
        if vectorizer is not None:
            self._vectorizer = vectorizer
        else:
            raise TypeError
        self._k = k
        if classifier is not None:
            self._classifier = classifier
        else:
            raise TypeError
        self._norm = None
        self._U, self._S, self._Vt = None, None, None

    def fit(self, X_train, y, more=None):
        tfidf = self._vectorizer.fit_transform(X_train).T
        self._U, self._S, self._Vt = svds(tfidf, k=self._k)
        self._S_inv = 1 / self._S
        self._y_train = y
        if more is None:
            self.tofit = self._Vt.T
            self.more = False
        else:
            self.tofit = np.column_stack((self._Vt.T, more))
            self.more = True
        self._classifier.fit(self.tofit, y)

    def predict(self, X, y=None, more=None):
        tfidf = self._vectorizer.transform(X)
        q_k = np.dot(tfidf.dot(self._U), np.diag(self._S_inv))
        if more is None:
            pred = self._classifier.predict(q_k)
        else:
            pred = self._classifier.predict(np.column_stack((q_k, more)))
        return pred

    def get_params(self, deep=True):
        return {'k': self._k, 'vectorizer': self._vectorizer, 'classifier': self._classifier}

    def set_params(self, **params):
        if 'k' in params:
            self._k = params['k']
        if 'vectorizer' in params:
            self._vectorizer = params['vectorizer']
        if 'classifier' in params:
            self._classifier = params['classifier']
            self._classifier.fit(self.tofit, self._y_train)

    def __copy__(self):
        newone = type(self)(self._vectorizer, self._classifier, self._k)
        newone.__dict__.update(self.__dict__)
        return newone

    """
    def fit_transform(self, X_train, y=None):
        self.fit(X_train)
        self.vt.T


    def transform(self, X, y=None):
        tfidf = self.vectorizer.transform(X)
        return np.dot(tfidf.dot(self.u), np.diag(self.s_inv))
    """