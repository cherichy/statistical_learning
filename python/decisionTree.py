import numpy as np
import pandas as pd


class DecisionTree:
    def __init__(self, data, method="ID3", eps=0.01):
        self.method = method
        self.eps = eps
        df = pd.DataFrame(data)
        self.tree = self.creatTree(df)

    def calH(self, count):
        # cal entropy
        p = count/count.sum()
        p = p[p != 0]
        return -p @ np.log2(p)

    def calg(self, cb):
        # cal gained entropy
        w = cb.sum(1)/cb.sum().sum()
        Hi = cb.apply(self.calH, axis=1)
        return w @ Hi

    def SplitData(self, df):
        labels = df.iloc[:, -1]
        data = df.iloc[:, :-1]
        # use crosstab to count the frequency
        cbs = (pd.crosstab(data.iloc[:, i], labels)
               for i in range(data.columns.size))
        y_c = labels.groupby(labels).count()
        # entropy of y
        HD = self.calH(y_c)
        HDA = [self.calg(cb) for cb in cbs]
        if self.method == "ID3":
            g = HD-HDA
        elif self.method == "C4.5":
            g = 1-HDA/HD
        if g.max() < self.eps:
            return None
        # the split location
        split = g.argmax()
        name = df.columns[split]
        # divide into parts
        gp = df.groupby(df.iloc[:, split])
        return ((name, i, d.drop(name, axis=1)) for i, d in gp)

    def creatTree(self, df):
        # all of one class
        if df.iloc[:, -1].unique().size == 1:
            return df.iloc[0, -1]
        if df.columns.size == 1:
            return df.mode().iloc[0, 0]
        res = {}
        # creat the tree recursively
        for name, i, d in self.SplitData(df):
            if name not in res:
                res[name] = {}
            res[name][i] = self.creatTree(d)
        return res


if __name__ == "__main__":
    df = pd.read_csv("../data/5-1.txt")
    t = DecisionTree(df)
    print(t.tree)
    df = pd.read_csv("../data/xigua4-1.txt")
    t = DecisionTree(df)
    print(t.tree)
