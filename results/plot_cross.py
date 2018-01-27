from matplotlib import pyplot as plt
import argparse
import numpy as np
import pandas as pd


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Synthesize results in cross folder")
    parser.add_argument('--no-plot', dest='plot', action='store_const', default=True, const=False)
    parser.add_argument('-v', '--verbose', dest='verbose', action='store_const', default=False, const=True)
    parser.add_argument('--metric', dest='metric', default='f_score')
    parser.add_argument('--best-matrix', action='store_const', default=False, const=True, dest='matrix')

    args = parser.parse_args()
    path = "./cross"
    ds = ["SarcasmCorpus", "data-sarc-sample", "wallace", "sarcasmv2"]
    dimensions = [40, 90, 140, 200, 250]
    classif = ["svm linear", "svm gaussian", "lr_L1", "lr_L2", "RF_gini", "RF_entropy", "DT_gini", "DT_entropy", "Bayes", "XGB"]
    colors = ['-r', '--r', '-b', '--b', '-g', '--g', '-k', '--k', '-m', '--m']
    toexclude = ["RF_gini", "RF_entropy", "DT_gini", "DT_entropy", "XGB"]

    best_matrix = []
    if args.matrix:
        best_matrix = np.zeros((len(ds), len(ds)))
    for i in range(len(ds)):
        scores = np.zeros((len(classif) - len(toexclude), len(dimensions)))
        for j in range(len(ds)):
            if i == j:
                continue
            for l in range(len(dimensions)):
                k = dimensions[l]
                file_path = path + str(k) + "/" + ds[i] + "-->" + ds[j] + '.csv'
                df = pd.read_csv(file_path, header=0)
                c = 0
                for model in classif:
                    if model in toexclude:
                        continue
                    scores[l, c] = df[df["model"]==model][args.metric]
                    c += 1

            if args.verbose:
                print("{0}-->{1}".format(ds[i], ds[j]))
                print(scores)


            if args.plot:
                plt.figure()
                for l in range(len(classif)):
                    if classif[l] in toexclude:
                        continue
                    plt.plot(k, scores[l, :], colors[l], label=classif[l])
                plt.title(ds[i] + "---" + ds[j])
                plt.legend(bbox_to_anchor=(1.0, 1.0), loc=2, borderaxespad=0.)
                plt.show()

            if args.matrix:
                best_matrix[i, j] = scores.max()

    if args.verbose:
        print("Best matrix:")
        print(ds)
        print(best_matrix)
