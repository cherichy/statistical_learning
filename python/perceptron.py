import numpy as np

class perceptron:
    def __init__(self,data,label,learning_rate=1):
        self.w=np.zeros(data.shape[1])
        self.b=0
        self.learning_rate=learning_rate
        self.X=data
        self.y=label
        
    def fit(self):
        num_wrong=self.X.shape[0]
        while num_wrong > 0:
            num_wrong = 0
            for x, y in zip(self.X,self.y):
                if y * (np.dot(x, self.w) + self.b) <= 0:
                    self.w += y * x * self.learning_rate
                    self.b += y * self.learning_rate
                    num_wrong += 1
            if num_wrong == 0:
                print("Fit finished!") 
    
    def fit_dual(self):
        a=np.zeros(data.shape[0])
        G=self.X @ self.X.T
        num_wrong=self.X.shape[0]
        while num_wrong > 0:
            num_wrong = 0
            for i, y in enumerate(self.y):
                if y * (self.b + np.dot(a * self.y, G[i])) <= 0:
                    a[i] += self.learning_rate
                    self.b += y * self.learning_rate
                    num_wrong += 1
            if num_wrong == 0:
                print("Fit finished!")
        self.w = ((a * self.y)[:,np.newaxis] * self.X).sum(0)
    
    def predict(self,data):
        res= ((data @ self.w.T + self.b)>0).astype(int)
        res[res==0]=-1
        return res

if __name__=="__main__":
    from pylab import *
    from sklearn.datasets import load_iris 
    # load data, use the iris dataset
    iris=load_iris()
    data=iris["data"][:100,:2]
    label=hstack((repeat(-1,50),repeat(1,50)))

    # build model, then fit
    model=perceptron(data,label,learning_rate=1)
    model.fit()
    w,b=model.w,model.b
    # plot the results
    scatter(*data[:50].T,label="-1")
    scatter(*data[50:].T,label="+1")
    xlabel("sepal length")
    ylabel("sepal width")
    legend()
    x=array([4,7.5])
    y=-(w[0]*x+b)/w[1]
    plot(x,y,"r-")
    show()
