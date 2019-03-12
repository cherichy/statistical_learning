import pandas as pd
import numpy as np


class NaiveBayes:
    def __init__(self, data, lam=0):
        df = pd.DataFrame(data)
        dim = df.shape[1]
        self.y_p = df[dim-1].groupby(df[dim-1]).count()+lam
        self.y_p /= self.y_p.sum()
        self.cb = []
        for i in range(dim-1):
            xi_p = pd.crosstab(df[i], df[dim-1])+lam
            self.cb.append(xi_p/xi_p.sum())

    def predict(self, x):
        labels = self.y_p.index
        res = pd.Series(np.zeros(labels.size), index=labels)
        for y in labels:
            res[y] = self.y_p[y]
            for i in range(len(x)):
                try:
                    res[y] *= self.cb[i][y][x[i]]
                except KeyError as e:
                    print('KeyError!', e)
                    return None
        print(res)
        return res.idxmax()


if __name__ == "__main__":
    df = pd.read_csv('../data/4-1.txt', header=None)
    NB = NaiveBayes(df, 1)
    res = NB.predict([2, 'S'])
    print('res:', res)
