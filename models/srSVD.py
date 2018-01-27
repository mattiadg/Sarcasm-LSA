# -*- coding: utf-8 -*-
"""
Created on Mon Feb 29 10:51:43 2016

@author: mattiadigangi
"""

from scipy.sparse.linalg import svds
import numpy as np

class SquareRootSVD:
    """This class creates a model that takes as input a text corpus,
    apply a vectorization algorithm to it, then computes a variant of
    TSVD as explained in "TSVD as a Statistical Estimator in the Latent Semantic
    Analysis Paradigm" by G. Pilato e G. Vassallo, 2014.
    In the end it applies a classifier to the vectors produced this way.
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
        
    def fit(self, texts, y, more=None):
        X = self._vectorizer.fit_transform(texts).T
        self._norm = X.sum()
        Q = X.multiply(1/self._norm)
        PSI_Q = Q.sqrt()
        self._U, self._S, self._Vt = svds(PSI_Q, k=self._k)
        self._y_train = y
        if more is None:
            self.tofit = self._Vt.T
            self.more = False
        else:
            self.tofit = np.column_stack((self._Vt.T, more))
            self.more = True
        self._S_inv = 1 / self._S
        self._classifier.fit(self.tofit, y)

    def predict(self, docs, y=None, more=None):
        if (self.more and more is None) or (not self.more and more is not None):
            raise TypeError
        q = self._vectorizer.transform(docs).T.multiply(1 / self._norm).sqrt()
        q_k = np.dot(q.T.dot(self._U), np.diag(self._S_inv))
        if more is None:
            pred = self._classifier.predict(q_k)
        else:
            pred = self._classifier.predict(np.column_stack((q_k, more)))

        return pred

    def get_params(self, deep=True):
        return {'k':self._k, 'vectorizer':self._vectorizer, 'classifier':self._classifier}
    
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
        def fit_transform(self, texts, y=None):
            self.fit(texts)
            return self.transform(texts)


        def transform(self, docs, y=None):
            s_inv = 1 / self._S
            q = self._vectorizer.transform(docs).T.multiply(1 / self._norm).sqrt()
            q_k = np.dot(q.T.dot(self._U), np.diag(s_inv))
            return q_k
        """