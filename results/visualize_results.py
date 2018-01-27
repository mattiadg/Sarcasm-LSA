# -*- coding: utf-8 -*-
"""
Created on Mon Jul 25 11:18:34 2016

@author: mattia

Questo script permette di visualizzare i risultati sull'esperimento lungo,
calcolando la media di una tra le tre misure precision, recall, f-score, 
mostrando come variano al variare della dimensionalità dei vettori.
Il grafico può essere riferito a tutti i modelli o ad uno solo.
"""

import argparse
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def correct_model_name(model):
    """Corregge il nome del modello per farlo corrispondere con quello del
        file csv"""
    if model in ['lr_L1', 'lr_L2', 'RF_entropy', 'DT_entropy', 'Bayes', 'XGB']:
        return model
    elif model in ['svm_linear', 'svm_gaussian']:
        return model.replace('_', ' ')


#####################################################################
###    Parsing degli argomenti                                    ###
#####################################################################
parser = argparse.ArgumentParser(description='Visualizza i risultati presenti \
        nella cartella new_results')
parser.add_argument('--exppath', dest='path', default='normal_experiment')
parser.add_argument('--models', dest='models', nargs='*', default=['all'])
parser.add_argument('--metric', dest='metric', default='f_score')
parser.add_argument('--dataset', dest='dataset', default='all')
parser.add_argument('--n-run', dest='run', default=10, type=int)
parser.add_argument('-v', '--verbose', dest='verbose', action='store_const',
                    default=False, const=True)
parser.add_argument('--no-plot', dest='plot', action='store_const',
                    default=True, const=False)
args = parser.parse_args()

######################################################################
###             Cattura degli argomenti                            ###
######################################################################
models = args.models
metric = args.metric
exppath = args.path
dataset = [args.dataset]
nruns = args.run

######################################################################
###             Validazione degli argomenti                        ###
######################################################################
valid_models = ['svm_linear', 'svm_gaussian', 'lr_L1', 'lr_L2', 'RF_entropy', \
                'DT_entropy', 'Bayes', 'XGB']
if models[0] != 'all':
    for model in models:
        if model not in valid_models:
            raise NameError(model + ' is not a valid model!')
else:
    models = valid_models

if args.models[0] != 'all' and not dataset[0] in ['all', 'filatova', 'data-sarc-sample', 'wallace', 'sarcasmv2',
                                                  'sarcasmv2_gen', 'sarcasmv2_hyp', 'sarcasmv2_rq']:
    raise NameError(dataset + ' is not a valid dataset!')
elif args.models[0] == 'all' and dataset[0] == 'all':
    dataset = ['filatova', 'data-sarc-sample', 'wallace', 'sarcasmv2', 'sarcasmv2_gen', 'sarcasmv2_hyp', 'sarcasmv2_rq']

######################################################################
###             Apertura dei file e visualizzazione                ###
######################################################################

path = './' + exppath + '/'

colors = ['r-', 'b-', 'k-', 'y-', 'm-', 'g-', 'c-', 'b--']
for ds in dataset:
    dir_path = path + ds + '/'
    metric_vals = np.zeros((len(models), nruns))
    if ds.startswith('sarcasmv2'):
        k_values = [40, 90, 140, 200, 250, 300, 350]
    else:
        k_values = [20, 40, 90, 120, 140, 170, 200, 220, 250, 280, 300, 350]
    to_plot = np.zeros((len(models), len(k_values)))
    j = 0
    for k in k_values:
        for run in range(1, nruns + 1):
            file_path = dir_path + 'run' + str(run) + 'k' + str(k)
            if ds == 'filatova':
                file_path += '_stars.csv'
            else:
                file_path += '.csv'
            # print(file_path)
            df = pd.read_csv(file_path, header=0, index_col=0)
            i = 0
            for model in models:
                metric_vals[i, run - 1] = df[df['model'] == correct_model_name(model)][metric].mean()
                i += 1
        to_plot[:, j] = np.mean(metric_vals, axis=1)
        j += 1

    if args.verbose:
        print(ds)
        print(models)
        print(to_plot)
    if args.plot:
        plt.figure()
        for l in range(len(models)):
            plt.plot(k_values, to_plot[l, :], colors[l], label=models[l])
        plt.xlabel('SVD dimension')
        plt.ylabel(metric)
        plt.title('experiment = ' + exppath + '; dataset = ' + ds)
        plt.legend(bbox_to_anchor=(1.0, 1.0), loc=2, borderaxespad=0.0)
        plt.show()
