import numpy as np
import scipy.optimize as so
from sklearn.datasets import load_iris

class BinLogistic:
    '''
    Binary logistic regression.
    '''
    def __init__(self,data,label):
        self.X=np.pad(data,((0,0),(0,1)),'constant',constant_values=1)
        self.y=label
        self.w=np.zeros(self.X.shape[1])
    def fit(self):
        def Loss(x,y,w):
            return - np.sum(y * (x @ w) - np.log( 1 + np.exp(x @ w)))
        def DLoss(x,y,w):
            t=np.exp(x @ w)
            return -(y - t / (1 + t)) @ x
        w0=np.zeros(3)
        f=lambda w:Loss(self.X,self.y,w)
        df=lambda w:DLoss(self.X,self.y,w)
        res=so.minimize(f,w0,method='BFGS',jac=df)
        self.w=res.x
    def predict(self,x):
        X=np.pad(x,((0,0),(0,1)),'constant',constant_values=1)
        return 1 if np.exp(X @ self.w)>1 else 0


class LogisticRegression:
    '''
    Multiple logistic regression. 
    An order of the label to be classified should be given for linear separability.
    '''
    def __init__(self,data,label,order=None):
        self.X=np.pad(data,((0,0),(0,1)),'constant',constant_values=1)
        self.y=label
        K=np.unique(label).shape[0]
        self.w=np.zeros((K-1,self.X.shape[1]))
        self.order = range(K) if order==None else order
    def fit(self):
        def Loss(x,y,w):
            return - np.sum(y * (x @ w) - np.log( 1 + np.exp(x @ w)))
        def DLoss(x,y,w):
            t=np.exp(x @ w)
            return -(y - t / (1 + t)) @ x
        w0=np.zeros(3)
        for index,i in enumerate(self.order[:-1]):
            label = (self.y==i).astype(int)
            f=lambda w:Loss(self.X,label,w)
            df=lambda w:DLoss(self.X,label,w)
            res=so.minimize(f,w0,method='BFGS',jac=df)
            self.w[index]=res.x
    def predict(self,x):
        X=np.pad(x,((0,0),(0,1)),'constant',constant_values=1)
        t=np.exp(X @ self.w.T)
        t=np.pad(t,((0,0),(0,1)),'constant',constant_values=1)
        res=np.zeros((t.shape[0],t.shape[1]+1))
        for index,i in enumerate(self.order):
            res[:,index]=t[:,i]
        return (res/res.sum(1)[:,np.newaxis]).argmax(1)
        

if __name__=="__main__":
    iris=load_iris()
    data=iris["data"][:100,:2]
    label=iris["target"][:100]
    model=LogisticRegression(data,label)
    model.fit()
    model.predict(data)

    m_data=iris["data"][:,[0,3]]
    m_label=iris["target"]
    MLR=LogisticRegression(m_data,m_label,[0,2,1])
    MLR.fit()
    acc=MLR.predict(m_data)-m_label

    from pylab import plot,show
    plot(*m_data[:50].T,'.')
    plot(*m_data[50:100].T,'.')
    plot(*m_data[100:150].T,'.')
    plot(*m_data[acc!=0].T,'x')
    show()

